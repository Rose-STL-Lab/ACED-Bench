"""
world_model.py

LLM-powered interface for translating natural language queries into
causal graph operations (observational and interventional sampling).

The World Model sits between the Scientist Agent and the causal simulator:
    1. Receives natural language queries from the Scientist Agent
    2. Uses an LLM to parse them into structured commands
    3. Validates commands against the causal graph structure
    4. Executes queries via the simulator (observational or do-interventional)
    5. Returns results in a structured format

The underlying causal graph uses do-calculus semantics:
    - Observational query: sample from the joint distribution P(V)
    - Interventional query: do(X=x) removes all incoming edges to X,
      fixing X's value and propagating the effect to X's descendants only.
      Parents of X are NOT affected by the intervention.

Usage:
    from bn_simulator import BNSimulator
    from bedrock_llm import BedrockLLM
    from world_model import WorldModel

    sim = BNSimulator.from_bif("world.bif")
    llm = BedrockLLM(model_id="us.anthropic.claude-opus-4-6-v1", max_new_tokens=512, temperature=0.1)
    world = WorldModel(simulator=sim, llm=llm, output_dir="./results")

    result = world.process_query("Give me 100 observational samples of smoke and cancer")
    result = world.process_query("Give me 200 samples where we intervene to set smoke=yes")
    print(result.to_xml())
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import pandas as pd

from schemas import (
    ParsedQuery,
    QueryType,
    QueryResult,
    WorldInfo,
    VariableInfo,
    QueryParseError,
    QueryValidationError,
    QueryExecutionError,
)
from bn_simulator import BNSimulator

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# World Model
# -----------------------------------------------------------------------------

@dataclass
class WorldModel:
    """
    LLM-powered interface between the Scientist Agent and the causal simulator.

    Responsibilities:
        - Parse natural language queries into structured observational or
          interventional commands
        - Validate commands against the causal graph structure
        - Execute queries via do-calculus: do(X=x) removes all incoming
          edges to X and propagates the fixed value to X's descendants
        - Format results for the Scientist Agent
    """
    simulator: BNSimulator
    llm: Any  # any object with generate(system_prompt, user_prompt)
    output_dir: str = "./query_results"

    # Configuration
    max_samples: int = 10000
    default_samples: int = 100
    preview_rows: int = 10
    max_total_samples: Optional[int] = None
    max_samples_per_query: Optional[int] = None
    sample_accounting: str = "rows"

    # Variable descriptions (semantic meanings)
    variable_descriptions: Dict[str, str] = field(default_factory=dict)

    # Story/context for the scientist
    story: str = ""

    # Non-intervenable variables: {var_name: reason}
    non_intervenable_variables: Dict[str, str] = field(default_factory=dict)

    # Internal state
    _query_counter: int = field(default=0, init=False)
    _sample_rows_used: int = field(default=0, init=False)
    _sample_cells_used: int = field(default=0, init=False)

    def __post_init__(self):
        out = Path(self.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(out)
        if self.sample_accounting not in ("rows", "cells"):
            raise ValueError("sample_accounting must be 'rows' or 'cells'")
        logger.info(f"WorldModel initialized. Output dir: {self.output_dir}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def process_query(self, query: str, seed: Optional[int] = None) -> QueryResult:
        """
        Process a natural language query from the Scientist Agent.

        Supports two query modes:
          - Observational: sample from the joint distribution P(V)
          - Interventional: do(X=x) — mutilate the causal graph by removing
            all incoming edges to X, fix X=x, then sample; X's parents are
            unaffected, X's descendants reflect the causal effect.

        Args:
            query: Natural language query
            seed: Optional random seed for reproducibility

        Returns:
            QueryResult with success status, data file path, and preview
        """
        self._query_counter += 1
        query_id = self._query_counter

        logger.info(f"Processing query #{query_id}: {query[:100]}...")

        try:
            parsed = self._parse_query(query)
            logger.info(f"Parsed: {parsed.query_type.value}, n={parsed.n_samples}")

            self._validate_query(parsed)
            self._validate_sample_budget(parsed)

            df = self._execute_query(parsed, seed=seed)
            self._record_sample_usage(df)

            result = self._create_success_result(parsed, df, query_id)
            return result

        except QueryParseError as e:
            logger.warning(f"Parse error: {e}")
            return self._create_error_result(query, str(e), "parse_error")

        except QueryValidationError as e:
            logger.warning(f"Validation error: {e}")
            return self._create_error_result(query, str(e), "validation_error")

        except QueryExecutionError as e:
            logger.error(f"Execution error: {e}")
            return self._create_error_result(query, str(e), "execution_error")

        except Exception as e:
            logger.exception("Unexpected error processing query")
            return self._create_error_result(query, f"Unexpected error: {e}", "unknown_error")

    def get_world_info(self) -> WorldInfo:
        """
        Get information about the causal system to present to the Scientist.

        Returns variable names, descriptions, and states — but NOT the causal
        graph structure. The scientist must discover causal relationships
        through querying.

        Returns:
            WorldInfo with story and variable descriptions (no graph structure)
        """
        variables = []
        for node in self.simulator.get_nodes():
            desc = self.variable_descriptions.get(node, "No description available")
            states = self.simulator.get_state_names(node)
            variables.append(VariableInfo(name=node, description=desc, states=states))

        if self.story:
            story = self.story
        else:
            story = (
                "You are investigating an unknown causal system. You can collect "
                "observational data (passive measurement) or perform interventions "
                "(do-calculus: forcibly set a variable to a value, severing its "
                "incoming causal links). Use these tools to discover which variables "
                "causally affect which others. The true causal graph is hidden."
            )

        return WorldInfo(
            story=story,
            variables=variables,
            non_intervenable_variables=self.non_intervenable_variables,
        )

    def set_story(self, story: str) -> None:
        """Set the story/context for this world."""
        self.story = story

    def set_variable_descriptions(self, descriptions: Dict[str, str]) -> None:
        """Set semantic descriptions for variables."""
        self.variable_descriptions = descriptions

    def set_non_intervenable_variables(self, non_intervenable: Dict[str, str]) -> None:
        """Set which variables cannot be intervened upon."""
        self.non_intervenable_variables = non_intervenable

    def reset_sample_usage(self) -> None:
        """Reset total sample accounting for a new question/experiment."""
        self._sample_rows_used = 0
        self._sample_cells_used = 0

    def get_sample_usage(self) -> Dict[str, Any]:
        """Return current cumulative sample usage."""
        units = (
            self._sample_rows_used
            if self.sample_accounting == "rows"
            else self._sample_cells_used
        )
        remaining = None
        if self.max_total_samples is not None:
            remaining = max(0, self.max_total_samples - units)
        return {
            "sample_rows_used": self._sample_rows_used,
            "sample_cells_used": self._sample_cells_used,
            "sample_units_used": units,
            "sample_accounting": self.sample_accounting,
            "max_total_samples": self.max_total_samples,
            "sample_units_remaining": remaining,
        }

    # -------------------------------------------------------------------------
    # Query Parsing (LLM-powered)
    # -------------------------------------------------------------------------

    def _parse_query(self, query: str) -> ParsedQuery:
        """Use LLM to parse natural language query into a structured command."""
        var_catalog = self._build_variable_catalog()
        system_prompt = self._get_parse_system_prompt()
        user_prompt = self._get_parse_user_prompt(query, var_catalog)

        llm_output = self.llm.generate(system_prompt, user_prompt)
        logger.debug(f"LLM parse output: {llm_output}")

        parsed_json = self._extract_json(llm_output, query)
        return self._json_to_parsed_query(parsed_json, query, llm_output)

    def _get_parse_system_prompt(self) -> str:
        """System prompt for query parsing."""
        return """You are a query parser for a causal discovery system. Your job is to convert natural language requests into \
structured JSON commands.

You MUST output ONLY a JSON object wrapped in <json></json> tags. No other text. What you put inside the tags must be valid JSON with {} braces.

The JSON schema is:
{
  "query_type": "observational" | "interventional",
  "n_samples": integer,
  "variables": null | ["var1", "var2", ...],
  "interventions": [] | [{"var1": "state1"}, {"var1": "state2"}, ...]
}

Rules:
- Use query_type="interventional" if the user mentions any of: "intervene", "intervention", "do()", "set", "force", "fix", "assign", \
"manipulate", "randomize to", "causally set"
- Use query_type="observational" if no intervention is mentioned (passive data collection)
- If user doesn't specify sample count, use n_samples=100
- If user doesn't specify which variables, use variables=null (meaning all)
- "interventions" is a list of intervention conditions; each element is one do() configuration (a dict mapping variable to state)
- Single condition example: [{"X": "a"}] — fix X to state a, sever X's incoming causal links
- Simultaneous interventions on multiple variables in ONE condition: [{"X": "a", "Y": "b"}]
- Multiple SEPARATE conditions (user asks for data under different do() values): [{"X": "a"}, {"X": "b"}] — n_samples will be drawn per condition
- Use exact variable and state names from the catalog provided
- An intervention do(X=x) fixes X to value x and severs X's incoming causal links"""

    def _get_parse_user_prompt(self, query: str, var_catalog: str) -> str:
        """User prompt for query parsing."""
        return f"""AVAILABLE VARIABLES AND STATES:
{var_catalog}

USER QUERY:
{query}

Parse this query into the JSON format. Output ONLY <json>...</json>."""

    def _build_variable_catalog(self) -> str:
        """Build a string listing all variables with their states."""
        lines = []
        for node in self.simulator.get_nodes():
            states = self.simulator.get_state_names(node)
            desc = self.variable_descriptions.get(node, "")
            desc_part = f" - {desc}" if desc else ""
            lines.append(f"- {node}: states=[{', '.join(states)}]{desc_part}")
        return "\n".join(lines)

    def _extract_json(self, llm_output: str, raw_query: str) -> Dict[str, Any]:
        """Extract JSON from LLM output."""
        match = re.search(r"<json>\s*(\{.*?\})\s*</json>", llm_output, re.DOTALL)

        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"\{[^{}]*\}", llm_output, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                raise QueryParseError(
                    "LLM output does not contain valid JSON",
                    raw_query=raw_query,
                    llm_output=llm_output,
                )

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise QueryParseError(
                f"Invalid JSON: {e}",
                raw_query=raw_query,
                llm_output=llm_output,
            ) from e

    def _json_to_parsed_query(
        self,
        data: Dict[str, Any],
        raw_query: str,
        llm_output: str,
    ) -> ParsedQuery:
        """Convert parsed JSON to ParsedQuery object."""
        query_type_str = data.get("query_type", "observational")
        if query_type_str not in ("observational", "interventional"):
            raise QueryParseError(
                f"Invalid query_type: {query_type_str}",
                raw_query=raw_query,
                llm_output=llm_output,
            )

        query_type = QueryType(query_type_str)

        n_samples = data.get("n_samples", self.default_samples)
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise QueryParseError(
                f"Invalid n_samples: {n_samples}",
                raw_query=raw_query,
                llm_output=llm_output,
            )

        variables = data.get("variables")
        if variables is not None and not isinstance(variables, list):
            raise QueryParseError(
                "variables must be null or a list",
                raw_query=raw_query,
                llm_output=llm_output,
            )

        raw_interventions = data.get("interventions", [])
        if isinstance(raw_interventions, dict):
            # a single condition given as a dict → single-element list
            interventions = [raw_interventions] if raw_interventions else []
        elif isinstance(raw_interventions, list):
            interventions = raw_interventions
        else:
            raise QueryParseError(
                "interventions must be an array of condition objects",
                raw_query=raw_query,
                llm_output=llm_output,
            )

        return ParsedQuery(
            query_type=query_type,
            n_samples=n_samples,
            variables=variables,
            interventions=interventions,
            raw_query=raw_query,
        )

    # -------------------------------------------------------------------------
    # Query Validation
    # -------------------------------------------------------------------------

    def _validate_query(self, parsed: ParsedQuery) -> None:
        """
        Validate parsed query against the causal graph structure.

        Checks:
            - Sample count within limits
            - Variables exist in the causal graph
            - Intervention states are valid
            - Non-intervenable variables are not intervened upon
            - Consistency (interventional requires interventions)
        """
        if parsed.n_samples > self.max_samples:
            raise QueryValidationError(
                f"Requested {parsed.n_samples} samples, maximum is {self.max_samples}"
            )

        valid_nodes = set(self.simulator.get_nodes())

        if parsed.variables is not None:
            unknown_vars = set(parsed.variables) - valid_nodes
            if unknown_vars:
                raise QueryValidationError(
                    f"Unknown variables: {sorted(unknown_vars)}. "
                    f"Available: {sorted(valid_nodes)}"
                )

        if parsed.query_type == QueryType.INTERVENTIONAL:
            if not parsed.interventions:
                raise QueryValidationError(
                    "Interventional query requires at least one intervention condition"
                )

            for condition in parsed.interventions:
                if not condition:
                    raise QueryValidationError(
                        "Each intervention condition must specify at least one variable"
                    )
                for var, state in condition.items():
                    if var not in valid_nodes:
                        raise QueryValidationError(
                            f"Unknown variable in intervention: '{var}'. "
                            f"Available: {sorted(valid_nodes)}"
                        )

                    if var in self.non_intervenable_variables:
                        reason = self.non_intervenable_variables[var]
                        raise QueryValidationError(
                            f"Cannot intervene on '{var}': {reason}. "
                            f"This variable is non-intervenable in this causal system."
                        )

                    valid_states = self.simulator.get_state_names(var)
                    if state not in valid_states:
                        raise QueryValidationError(
                            f"Invalid state '{state}' for variable '{var}'. "
                            f"Valid states: {valid_states}"
                        )

        elif parsed.interventions:
            raise QueryValidationError(
                "Observational query should not have interventions"
            )

    def _estimate_requested_sample_usage(self, parsed: ParsedQuery) -> Dict[str, int]:
        """Estimate rows/cells before execution for budget validation."""
        n_conditions = (
            len(parsed.interventions)
            if parsed.query_type == QueryType.INTERVENTIONAL
            else 1
        )
        rows = parsed.n_samples * n_conditions
        n_variables = (
            len(parsed.variables)
            if parsed.variables is not None
            else len(self.simulator.get_nodes())
        )
        cells = rows * n_variables
        units = rows if self.sample_accounting == "rows" else cells
        return {"rows": rows, "cells": cells, "units": units}

    def _validate_sample_budget(self, parsed: ParsedQuery) -> None:
        """Reject queries that exceed per-query or remaining total sample budgets."""
        requested = self._estimate_requested_sample_usage(parsed)

        if self.max_samples_per_query is not None and requested["units"] > self.max_samples_per_query:
            raise QueryValidationError(
                "Sample budget exceeded: requested "
                f"{requested['units']} {self.sample_accounting} in one query, "
                f"per-query limit is {self.max_samples_per_query}."
            )

        if self.max_total_samples is None:
            return

        used = (
            self._sample_rows_used
            if self.sample_accounting == "rows"
            else self._sample_cells_used
        )
        remaining = max(0, self.max_total_samples - used)
        if requested["units"] > remaining:
            raise QueryValidationError(
                "Sample budget exceeded: requested "
                f"{requested['units']} {self.sample_accounting}, only "
                f"{remaining} {self.sample_accounting} remain. Ask for <= "
                f"{remaining} {self.sample_accounting} or answer from existing data."
            )

    def _record_sample_usage(self, df: pd.DataFrame) -> None:
        """Account by actual returned rows/cells after successful execution."""
        rows = len(df)
        data_columns = [c for c in df.columns if c != "__intervention__"]
        cells = rows * len(data_columns)
        self._sample_rows_used += rows
        self._sample_cells_used += cells

    # -------------------------------------------------------------------------
    # Query Execution
    # -------------------------------------------------------------------------

    def _execute_query(
        self,
        parsed: ParsedQuery,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Execute the parsed query against the causal simulator.

        Observational: sample from P(V).
        Interventional: mutilate the graph with do(X=x) — remove all incoming
        edges to X, fix X to the specified state, then sample. X's parents
        remain unaffected; X's descendants reflect the causal effect.
        """
        try:
            if parsed.query_type == QueryType.OBSERVATIONAL:
                df = self.simulator.sample_observational(
                    n=parsed.n_samples,
                    variables=parsed.variables,
                    seed=seed,
                )
            else:
                dfs = []
                multi = len(parsed.interventions) > 1
                for condition in parsed.interventions:
                    df_c = self.simulator.sample_interventional(
                        interventions=condition,
                        n=parsed.n_samples,
                        variables=parsed.variables,
                        seed=seed,
                    )
                    if multi:
                        label = "do(" + ", ".join(f"{k}={v}" for k, v in condition.items()) + ")"
                        df_c["__intervention__"] = label
                    dfs.append(df_c)
                df = pd.concat(dfs, ignore_index=True)
            return df

        except Exception as e:
            raise QueryExecutionError(f"Sampling failed: {e}") from e

    # -------------------------------------------------------------------------
    # Result Creation
    # -------------------------------------------------------------------------

    def _create_success_result(
        self,
        parsed: ParsedQuery,
        df: pd.DataFrame,
        query_id: int,
    ) -> QueryResult:
        """Create a successful QueryResult."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"query_{query_id:04d}_{parsed.query_type.value}_{timestamp}.csv"
        filepath = Path(self.output_dir) / filename

        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} samples to {filepath}")

        preview = df.head(self.preview_rows).to_csv(index=False)

        return QueryResult(
            success=True,
            query=parsed,
            data_file=str(filepath),
            n_rows=len(df),
            columns=list(df.columns),
            preview=preview,
            sample_rows=len(df),
            sample_cells=len(df) * len([c for c in df.columns if c != "__intervention__"]),
            sample_units=(
                len(df)
                if self.sample_accounting == "rows"
                else len(df) * len([c for c in df.columns if c != "__intervention__"])
            ),
            sample_accounting=self.sample_accounting,
            sample_usage_after=self.get_sample_usage(),
        )

    def _create_error_result(
        self,
        raw_query: str,
        error_message: str,
        error_type: str,
    ) -> QueryResult:
        """Create an error QueryResult."""
        parsed = ParsedQuery(
            query_type=QueryType.OBSERVATIONAL,
            n_samples=0,
            variables=None,
            interventions=[],
            raw_query=raw_query,
        )

        return QueryResult(
            success=False,
            query=parsed,
            error_message=f"[{error_type}] {error_message}",
            sample_accounting=self.sample_accounting,
            sample_usage_after=self.get_sample_usage(),
        )
