"""
schemas.py

Shared data structures and message formats for the causal discovery system.
All inter-agent communication uses these schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class QueryType(Enum):
    """Type of query the scientist can make."""
    OBSERVATIONAL = "observational"
    INTERVENTIONAL = "interventional"


@dataclass
class ParsedQuery:
    """
    Structured representation of a scientist's query.
    
    Attributes:
        query_type: observational or interventional
        n_samples: number of samples requested
        variables: which variables to include (None = all)
        interventions: dict of {variable: state} for do() operations
        raw_query: the original natural language query
    """
    query_type: QueryType
    n_samples: int
    variables: Optional[List[str]]
    interventions: List[Dict[str, str]]  # list of intervention conditions; each dict is one do() config
    raw_query: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "n_samples": self.n_samples,
            "variables": self.variables,
            "interventions": self.interventions,
            "raw_query": self.raw_query,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParsedQuery":
        raw = d.get("interventions", [])
        if isinstance(raw, dict):
            # a single condition given as a dict
            interventions = [raw] if raw else []
        else:
            interventions = raw
        return cls(
            query_type=QueryType(d["query_type"]),
            n_samples=d["n_samples"],
            variables=d.get("variables"),
            interventions=interventions,
            raw_query=d.get("raw_query", ""),
        )


@dataclass
class QueryResult:
    """
    Result of a query execution.
    
    Attributes:
        success: whether the query executed successfully
        query: the parsed query that was executed
        data_file: path to CSV file with samples (if successful)
        n_rows: number of rows in the result
        columns: list of column names
        preview: first few rows as string (for quick inspection)
        error_message: error description (if failed)
    """
    success: bool
    query: ParsedQuery
    data_file: Optional[str] = None
    n_rows: int = 0
    columns: List[str] = field(default_factory=list)
    preview: str = ""
    error_message: Optional[str] = None
    sample_rows: int = 0
    sample_cells: int = 0
    sample_units: int = 0
    sample_accounting: str = "rows"
    sample_usage_after: Optional[Dict[str, Any]] = None
    
    def to_xml(self) -> str:
        """Format result as XML for LLM consumption."""
        lines = ["<query_result>"]
        lines.append(f"  <success>{str(self.success).lower()}</success>")
        lines.append(f"  <query_type>{self.query.query_type.value}</query_type>")
        
        if self.query.interventions:
            cond_strs = [
                "do(" + ", ".join(f"{k}={v}" for k, v in c.items()) + ")"
                for c in self.query.interventions
            ]
            lines.append(f"  <interventions>{' | '.join(cond_strs)}</interventions>")
        
        if self.success:
            lines.append(f"  <n_samples>{self.n_rows}</n_samples>")
            lines.append(f"  <columns>{', '.join(self.columns)}</columns>")
            if self.sample_usage_after:
                remaining = self.sample_usage_after.get("sample_units_remaining")
                remaining_text = "unlimited" if remaining is None else str(remaining)
                lines.append(
                    "  <sample_budget>"
                    f"{self.sample_usage_after.get('sample_units_used', 0)} "
                    f"{self.sample_usage_after.get('sample_accounting', self.sample_accounting)} used; "
                    f"{remaining_text} remaining"
                    "</sample_budget>"
                )
            lines.append(f"  <data_file>{self.data_file}</data_file>")
            lines.append(f"  <preview>\n{self.preview}\n  </preview>")
        else:
            lines.append(f"  <error>{self.error_message}</error>")
        
        lines.append("</query_result>")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "query": self.query.to_dict(),
            "data_file": self.data_file,
            "n_rows": self.n_rows,
            "columns": self.columns,
            "preview": self.preview,
            "error_message": self.error_message,
            "sample_rows": self.sample_rows,
            "sample_cells": self.sample_cells,
            "sample_units": self.sample_units,
            "sample_accounting": self.sample_accounting,
            "sample_usage_after": self.sample_usage_after,
        }


@dataclass
class VariableInfo:
    """Information about a variable in the causal system."""
    name: str
    description: str
    states: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "states": self.states,
        }


@dataclass
class WorldInfo:
    """
    Information about the world that is revealed to the scientist.
    This does NOT include the graph structure - that's what they must discover.
    """
    story: str
    variables: List[VariableInfo]
    non_intervenable_variables: Dict[str, str] = field(default_factory=dict)
    # Maps variable name -> reason why it cannot be intervened upon

    def to_xml(self) -> str:
        """Format world info as XML for the scientist."""
        lines = ["<world_info>"]
        lines.append(f"  <story>{self.story}</story>")
        lines.append("  <variables>")
        for var in self.variables:
            lines.append(f"    <variable>")
            lines.append(f"      <name>{var.name}</name>")
            lines.append(f"      <description>{var.description}</description>")
            lines.append(f"      <states>{', '.join(var.states)}</states>")
            if var.name in self.non_intervenable_variables:
                reason = self.non_intervenable_variables[var.name]
                lines.append(f"      <intervenable>no</intervenable>")
                lines.append(f"      <non_intervenable_reason>{reason}</non_intervenable_reason>")
            else:
                lines.append(f"      <intervenable>yes</intervenable>")
            lines.append(f"    </variable>")
        lines.append("  </variables>")
        lines.append("</world_info>")
        return "\n".join(lines)

    def get_variable_catalog(self) -> str:
        """Get a compact string listing all variables."""
        lines = []
        for var in self.variables:
            states_str = ", ".join(var.states)
            interv = " [NON-INTERVENABLE]" if var.name in self.non_intervenable_variables else ""
            lines.append(f"- {var.name}: {var.description} (states: {states_str}){interv}")
        return "\n".join(lines)


@dataclass
class Question:
    """A question posed to the scientist."""
    question_type: str  # e.g., "direct_edge", "d_separation", etc.
    question_text: str
    ground_truth: Any  # The correct answer
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_xml(self) -> str:
        return f"""<question>
  <type>{self.question_type}</type>
  <text>{self.question_text}</text>
</question>"""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_type": self.question_type,
            "question_text": self.question_text,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# Error types
# -----------------------------------------------------------------------------

class WorldModelError(Exception):
    """Base error for world model failures."""
    pass


class QueryParseError(WorldModelError):
    """Failed to parse the scientist's query."""

    def __init__(self, message: str, raw_query: str, llm_output: str = ""):
        super().__init__(message, raw_query, llm_output)
        self.message = message
        self.raw_query = raw_query
        self.llm_output = llm_output


class QueryValidationError(WorldModelError):
    """Query parsed but is invalid (bad variable names, states, etc.)."""
    pass


class QueryExecutionError(WorldModelError):
    """Query valid but execution failed."""
    pass
