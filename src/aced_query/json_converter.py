"""
json_converter.py

Converts JSON Bayesian Network definitions to BIF format for use with pgmpy.

JSON Format Expected:
{
  "meta": { ... },
  "variables": [
    {"name": "X", "values": ["a", "b"], "desc": "description"},
    ...
  ],
  "edges": [
    ["parent", "child"],
    ...
  ],
  "cpds": [
    {
      "child": "X",
      "parents": ["P1", "P2"],
      "values": [[...], [...], ...],  # One row per state of child
      "cardinality": N
    },
    ...
  ],
  "questions": [...]  # Optional
}

CPD Values Format:
- Each inner list is one row (one state of the child variable)
- Columns correspond to all combinations of parent states
- Parent state combinations are in lexicographic order (first parent varies slowest)
- For root nodes (no parents), each row has a single value

Usage:
    from json_converter import JSONToBIFConverter
    
    converter = JSONToBIFConverter("world.json")
    converter.convert("world.bif")
    
    # Or get the config for the system
    config = converter.get_world_config()
"""

from __future__ import annotations

import json
import re
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field


def _sanitize_bif_name(s: str) -> str:
    """Sanitize a string for use as a BIF identifier (network name or state name).

    BIF identifiers must be alphanumeric + underscores.  This function
    replaces common special characters with readable equivalents, then
    strips anything left over.
    """
    s = s.replace(">=", "gte").replace("<=", "lte")
    s = s.replace("≥", "gte").replace("≤", "lte")
    s = s.replace(">", "gt").replace("<", "lt")
    s = s.replace("&", "and")
    s = s.replace("/", "_").replace("\\", "_")
    s = s.replace(" ", "_").replace("-", "_")
    s = s.replace("(", "_").replace(")", "_")
    s = s.replace("'", "").replace('"', "")
    # Remove any remaining non-alphanumeric/underscore characters
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    # Collapse multiple underscores and strip leading/trailing
    s = re.sub(r"_+", "_", s).strip("_")
    # BIF identifiers cannot start with a digit
    if s and s[0].isdigit():
        s = "v" + s
    return s or "unknown"


@dataclass
class JSONToBIFConverter:
    """Converts JSON Bayesian Network format to BIF format."""
    
    json_path: str
    
    # Parsed data
    data: Dict[str, Any] = field(default=None, init=False)
    variables: List[Dict] = field(default_factory=list, init=False)
    edges: List[List[str]] = field(default_factory=list, init=False)
    cpds: List[Dict] = field(default_factory=list, init=False)
    questions: List[Dict] = field(default_factory=list, init=False)
    meta: Dict[str, Any] = field(default_factory=dict, init=False)
    story: str = field(default="", init=False)
    non_intervenable_variables: List[Dict] = field(default_factory=list, init=False)
    
    # Derived structures
    _var_to_states: Dict[str, List[str]] = field(default_factory=dict, init=False)
    _parents_map: Dict[str, List[str]] = field(default_factory=dict, init=False)
    _cpd_map: Dict[str, Dict] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self._load_json()
        self._build_indices()
    
    def _load_json(self) -> None:
        """Load and parse the JSON file."""
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        self.meta = self.data.get("meta", {})
        self.variables = self.data.get("variables", [])
        self.edges = self.data.get("edges", [])
        self.cpds = self.data.get("cpds", [])
        self.questions = self.data.get("questions", [])
        self.story = self.data.get("story", "")
        self.non_intervenable_variables = self.data.get("non_intervenable_variables", [])
    
    def _build_indices(self) -> None:
        """Build lookup indices for efficient access."""
        # Variable name -> states mapping
        for var in self.variables:
            self._var_to_states[var["name"]] = var["values"]

        # CPD lookup by child name
        for cpd in self.cpds:
            self._cpd_map[cpd["child"]] = cpd

        # Build parents map.  IMPORTANT: parent order MUST come from the CPD
        # spec, not from edge-list order.  pgmpy lays out `cpd.values`
        # column-by-column using the order it was given as `evidence=`, and
        # the world files store that order in `cpd["parents"]`.
        # Iterating parents in any other order (e.g. edge-insertion order)
        # silently transposes the CPD when we enumerate parent combinations
        # for the BIF probability table — producing samples from a different
        # distribution than the JSON's gold metadata describes.
        for var in self.variables:
            cpd = self._cpd_map.get(var["name"])
            if cpd is not None:
                self._parents_map[var["name"]] = list(cpd.get("parents", []))
            else:
                self._parents_map[var["name"]] = []
    
    def convert(self, output_path: str) -> str:
        """
        Convert JSON to BIF format and save to file.
        
        Args:
            output_path: Path for output BIF file
            
        Returns:
            The BIF content as a string
        """
        bif_content = self._generate_bif()
        
        with open(output_path, 'w') as f:
            f.write(bif_content)
        
        return bif_content
    
    def _generate_bif(self) -> str:
        """Generate BIF format content."""
        lines = []
        
        # Header
        network_name = _sanitize_bif_name(self.meta.get("topic", "network"))
        lines.append(f"// Bayesian Network: {self.meta.get('topic', 'Unknown')}")
        lines.append(f"// Nodes: {self.meta.get('n_nodes', len(self.variables))}")
        lines.append(f"// Edges: {len(self.edges)}")
        lines.append(f"// Generated from: {Path(self.json_path).name}")
        lines.append("")
        lines.append(f"network {network_name} {{}}")
        lines.append("")
        
        # Variable declarations
        for var in self.variables:
            lines.extend(self._format_variable(var))
            lines.append("")
        
        # Probability tables
        for var in self.variables:
            lines.extend(self._format_probability(var["name"]))
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_variable(self, var: Dict) -> List[str]:
        """Format a variable declaration."""
        name = var["name"]
        states = var["values"]
        
        safe_states = [_sanitize_bif_name(s) for s in states]
        states_str = ", ".join(safe_states)
        
        return [
            f"variable {name} {{",
            f"  type discrete [ {len(states)} ] {{ {states_str} }};",
            f"}}"
        ]
    
    def _format_probability(self, var_name: str) -> List[str]:
        """Format a probability table for a variable."""
        parents = self._parents_map.get(var_name, [])
        cpd = self._cpd_map.get(var_name)
        
        if cpd is None:
            raise ValueError(f"No CPD found for variable '{var_name}'")
        
        lines = []
        
        # Header
        if parents:
            parents_str = ", ".join(parents)
            lines.append(f"probability ( {var_name} | {parents_str} ) {{")
        else:
            lines.append(f"probability ( {var_name} ) {{")
        
        # Values
        values = cpd["values"]  # List of rows, one per child state
        
        if not parents:
            # Root node: simple table
            # values is [[p1], [p2], ...] - one single value per state
            probs = [row[0] for row in values]
            probs_str = ", ".join(f"{p:.10f}" for p in probs)
            lines.append(f"  table {probs_str};")
        else:
            # Node with parents: conditional table
            # Generate all parent state combinations
            parent_states_list = [self._var_to_states[p] for p in parents]
            parent_combos = list(itertools.product(*parent_states_list))
            
            # Each column in values corresponds to one parent combination
            n_child_states = len(values)
            
            for col_idx, combo in enumerate(parent_combos):
                # Get probabilities for this parent combination (column)
                probs = [values[row_idx][col_idx] for row_idx in range(n_child_states)]
                
                safe_combo = [_sanitize_bif_name(s) for s in combo]
                combo_str = ", ".join(safe_combo)
                probs_str = ", ".join(f"{p:.10f}" for p in probs)
                
                lines.append(f"  ({combo_str}) {probs_str};")
        
        lines.append("}")
        
        return lines
    
    def get_world_config(self) -> Dict[str, Any]:
        """
        Extract configuration for the causal discovery system.
        
        Returns:
            Dict with variable_descriptions, story, questions, etc.
        """
        # Build variable descriptions
        var_descriptions = {}
        for var in self.variables:
            var_descriptions[var["name"]] = var.get("desc", "No description")

        # Use story from JSON if available, otherwise generate a generic one
        topic = self.meta.get("topic", "Unknown Domain")
        n_nodes = self.meta.get("n_nodes", len(self.variables))

        if self.story:
            story = self.story
        else:
            story = (
                f"You are a researcher investigating causal relationships in the domain of {topic}. "
                f"This system contains {n_nodes} variables representing various factors in this domain. "
                f"Your goal is to understand the causal structure underlying these observations."
            )

        # Build non-intervenable dict: {var_name: reason}
        non_intervenable = {
            item["name"]: item.get("reason", "Cannot be manipulated")
            for item in self.non_intervenable_variables
        }

        # Format questions
        formatted_questions = []
        for q in self.questions:
            formatted_questions.append({
                "question_type": q.get("question_type", "unknown"),
                "question_text": q.get("question", ""),
                "ground_truth": q.get("answer", ""),
                "explanation": q.get("explanation", ""),
                "difficulty": q.get("difficulty", "unknown"),
                "relevant_variables": q.get("relevant_variables", []),
                "id": q.get("id", 0),
            })

        return {
            "name": self.meta.get("topic", "unknown").replace(" ", "_").lower(),
            "topic": self.meta.get("topic", "Unknown"),
            "variable_descriptions": var_descriptions,
            "story": story,
            "non_intervenable_variables": non_intervenable,
            "questions": formatted_questions,
            "n_nodes": n_nodes,
            "n_edges": len(self.edges),
            "edges": self.edges,  # Ground truth for evaluation
        }
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """Return list of edges as tuples."""
        return [(e[0], e[1]) for e in self.edges]
    
    def validate(self) -> List[str]:
        """
        Validate the JSON structure for conversion.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if not self.variables:
            errors.append("No variables defined")
        
        if not self.cpds:
            errors.append("No CPDs defined")
        
        # Check all variables have CPDs
        var_names = set(v["name"] for v in self.variables)
        cpd_vars = set(c["child"] for c in self.cpds)
        
        missing_cpds = var_names - cpd_vars
        if missing_cpds:
            errors.append(f"Variables missing CPDs: {missing_cpds}")
        
        extra_cpds = cpd_vars - var_names
        if extra_cpds:
            errors.append(f"CPDs for unknown variables: {extra_cpds}")
        
        # Check edges reference valid variables
        for parent, child in self.edges:
            if parent not in var_names:
                errors.append(f"Edge references unknown parent: {parent}")
            if child not in var_names:
                errors.append(f"Edge references unknown child: {child}")
        
        # Check CPD dimensions
        for cpd in self.cpds:
            child = cpd["child"]
            parents = cpd.get("parents", [])
            values = cpd["values"]
            
            # Check number of rows matches cardinality
            expected_rows = cpd.get("cardinality", len(self._var_to_states.get(child, [])))
            if len(values) != expected_rows:
                errors.append(
                    f"CPD for {child}: expected {expected_rows} rows, got {len(values)}"
                )
            
            # Check number of columns
            if parents:
                expected_cols = 1
                for p in parents:
                    expected_cols *= len(self._var_to_states.get(p, []))
                
                for row_idx, row in enumerate(values):
                    if len(row) != expected_cols:
                        errors.append(
                            f"CPD for {child}, row {row_idx}: "
                            f"expected {expected_cols} columns, got {len(row)}"
                        )
            
            # Check probabilities sum to 1 (approximately)
            if values and values[0]:
                n_cols = len(values[0])
                for col_idx in range(n_cols):
                    col_sum = sum(values[row_idx][col_idx] for row_idx in range(len(values)))
                    if abs(col_sum - 1.0) > 0.01:
                        errors.append(
                            f"CPD for {child}, column {col_idx}: "
                            f"probabilities sum to {col_sum:.4f}, expected 1.0"
                        )
        
        return errors


def convert_json_to_bif(json_path: str, bif_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to convert JSON to BIF.
    
    Args:
        json_path: Path to input JSON file
        bif_path: Path for output BIF file
        
    Returns:
        Tuple of (BIF content, world config dict)
    """
    converter = JSONToBIFConverter(json_path)
    
    # Validate first
    errors = converter.validate()
    if errors:
        raise ValueError(f"JSON validation failed:\n" + "\n".join(errors))
    
    # Convert
    bif_content = converter.convert(bif_path)
    config = converter.get_world_config()
    
    return bif_content, config
