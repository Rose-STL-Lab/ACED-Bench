"""
bn_simulator.py

A clean wrapper around pgmpy for Bayesian Network simulation.

Provides:
    - Loading from BIF files
    - Observational sampling
    - Interventional sampling (do-calculus)
    - Basic introspection (nodes, edges, states)

Usage:
    sim = BNSimulator.from_bif("world.bif")
    df_obs = sim.sample_observational(n=100)
    df_int = sim.sample_interventional({"smoke": "yes"}, n=100)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from copy import deepcopy
from pathlib import Path
import logging

import pandas as pd
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork

logger = logging.getLogger(__name__)


@dataclass
class BNSimulator:
    """
    Simulator for discrete Bayesian Networks.
    
    Wraps a pgmpy BayesianNetwork and provides clean sampling APIs.
    """
    
    model: BayesianNetwork
    name: str = "bn_world"
    
    # Cached metadata (populated in __post_init__)
    _state_names: Dict[str, List[str]] = field(default_factory=dict, repr=False)
    _cardinality: Dict[str, int] = field(default_factory=dict, repr=False)
    _sampler: BayesianModelSampling = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate model and cache metadata."""
        self._validate_and_cache()
    
    def _validate_and_cache(self) -> None:
        """Validate the model and cache state information."""
        # Validate model structure
        try:
            self.model.check_model()
        except Exception as e:
            raise ValueError(f"Invalid Bayesian Network model: {e}") from e
        
        # Cache state names and cardinalities for each node
        self._state_names = {}
        self._cardinality = {}
        
        for node in self.model.nodes():
            cpd = self.model.get_cpds(node)
            if cpd is None:
                raise ValueError(f"Node '{node}' has no CPD defined.")
            
            # Get state names from CPD
            if node in cpd.state_names:
                self._state_names[node] = list(cpd.state_names[node])
            else:
                # Fallback: generate numeric state names
                self._state_names[node] = [str(i) for i in range(cpd.variable_card)]
            
            self._cardinality[node] = len(self._state_names[node])
        
        # Initialize sampler
        self._sampler = BayesianModelSampling(self.model)
        
        logger.info(
            f"BNSimulator initialized: {len(self.get_nodes())} nodes, "
            f"{len(self.get_edges())} edges"
        )
    
    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------
    
    @classmethod
    def from_bif(cls, bif_path: str, name: Optional[str] = None) -> "BNSimulator":
        """
        Create a BNSimulator from a .bif file.
        
        Args:
            bif_path: Path to the BIF file
            name: Optional name for the simulator (defaults to filename)
            
        Returns:
            BNSimulator instance
            
        Raises:
            FileNotFoundError: If BIF file doesn't exist
            ValueError: If BIF file is invalid
        """
        path = Path(bif_path)
        if not path.exists():
            raise FileNotFoundError(f"BIF file not found: {bif_path}")
        
        try:
            reader = BIFReader(str(path))
            model = reader.get_model()
        except Exception as e:
            raise ValueError(f"Failed to parse BIF file '{bif_path}': {e}") from e
        
        sim_name = name or path.stem
        return cls(model=model, name=sim_name)
    
    # -------------------------------------------------------------------------
    # Sampling API
    # -------------------------------------------------------------------------
    
    def sample_observational(
        self,
        n: int,
        variables: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Draw observational samples from the joint distribution.
        
        Args:
            n: Number of i.i.d. samples to draw
            variables: Subset of variables to return (None = all)
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with n rows, one column per variable
            
        Raises:
            ValueError: If n <= 0 or variables contain unknown names
        """
        self._validate_sample_request(n, variables)
        
        df = self._sampler.forward_sample(size=n, seed=seed)
        
        if variables is not None:
            df = df[list(variables)]
        
        return df.reset_index(drop=True)
    
    def sample_interventional(
        self,
        interventions: Dict[str, str],
        n: int,
        variables: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Draw samples under intervention do(X=x).
        
        Implements the do-operator by:
            1. Removing all incoming edges to intervened variables
            2. Setting their distributions to point masses
        
        Args:
            interventions: Dict of {variable: state} specifying do() operations
            n: Number of samples to draw
            variables: Subset of variables to return (None = all)
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with n rows sampled from the mutilated graph
            
        Raises:
            ValueError: If interventions reference unknown variables/states
        """
        self._validate_sample_request(n, variables)
        self._validate_interventions(interventions)
        
        # Create mutilated model
        mutilated_model = self._apply_interventions(interventions)
        mutilated_sampler = BayesianModelSampling(mutilated_model)
        
        df = mutilated_sampler.forward_sample(size=n, seed=seed)
        
        if variables is not None:
            df = df[list(variables)]
        
        return df.reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # Intervention Implementation
    # -------------------------------------------------------------------------
    
    def _apply_interventions(
        self,
        interventions: Dict[str, str],
    ) -> BayesianNetwork:
        """
        Create a mutilated graph with interventions applied.
        
        Does NOT modify the original model.
        
        Args:
            interventions: Dict of {variable: state}
            
        Returns:
            New BayesianNetwork with interventions applied
        """
        mutilated = deepcopy(self.model)
        
        for var, state in interventions.items():
            # Remove all incoming edges
            parents = list(mutilated.get_parents(var))
            for parent in parents:
                mutilated.remove_edge(parent, var)
            
            # Replace CPD with point mass at intervened state
            old_cpd = mutilated.get_cpds(var)
            states = self._state_names[var]
            state_idx = states.index(state)
            
            # Create delta distribution: P(var=state) = 1, all others = 0
            values = [[0.0] for _ in range(len(states))]
            values[state_idx] = [1.0]
            
            new_cpd = TabularCPD(
                variable=var,
                variable_card=len(states),
                values=values,
                state_names={var: states},
            )
            
            mutilated.remove_cpds(old_cpd)
            mutilated.add_cpds(new_cpd)
        
        # Validate the mutilated model
        mutilated.check_model()
        
        return mutilated
    
    # -------------------------------------------------------------------------
    # Validation Helpers
    # -------------------------------------------------------------------------
    
    def _validate_sample_request(
        self,
        n: int,
        variables: Optional[Sequence[str]],
    ) -> None:
        """Validate sampling parameters."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        
        if variables is not None:
            unknown = set(variables) - set(self.get_nodes())
            if unknown:
                raise ValueError(
                    f"Unknown variables: {sorted(unknown)}. "
                    f"Available: {sorted(self.get_nodes())}"
                )
    
    def _validate_interventions(
        self,
        interventions: Dict[str, str],
    ) -> None:
        """Validate intervention specification."""
        if not interventions:
            raise ValueError("Interventions dict cannot be empty for interventional query")
        
        for var, state in interventions.items():
            if var not in self._state_names:
                raise ValueError(
                    f"Unknown variable in intervention: '{var}'. "
                    f"Available: {sorted(self.get_nodes())}"
                )
            
            valid_states = self._state_names[var]
            if state not in valid_states:
                raise ValueError(
                    f"Invalid state '{state}' for variable '{var}'. "
                    f"Valid states: {valid_states}"
                )
    
    # -------------------------------------------------------------------------
    # Introspection API
    # -------------------------------------------------------------------------
    
    def get_nodes(self) -> List[str]:
        """Return list of all variable names."""
        return list(self.model.nodes())
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """Return list of directed edges as (parent, child) tuples."""
        return list(self.model.edges())
    
    def get_state_names(self, variable: str) -> List[str]:
        """
        Return list of state names for a variable.
        
        Raises:
            KeyError: If variable doesn't exist
        """
        if variable not in self._state_names:
            raise KeyError(f"Unknown variable: '{variable}'")
        return self._state_names[variable]
    
    def get_cardinality(self, variable: str) -> int:
        """
        Return the number of states for a variable.
        
        Raises:
            KeyError: If variable doesn't exist
        """
        if variable not in self._cardinality:
            raise KeyError(f"Unknown variable: '{variable}'")
        return self._cardinality[variable]
    
    def get_parents(self, variable: str) -> List[str]:
        """Return list of parent nodes for a variable."""
        if variable not in self.model.nodes():
            raise KeyError(f"Unknown variable: '{variable}'")
        return list(self.model.get_parents(variable))
    
    def get_children(self, variable: str) -> List[str]:
        """Return list of child nodes for a variable."""
        if variable not in self.model.nodes():
            raise KeyError(f"Unknown variable: '{variable}'")
        return list(self.model.get_children(variable))
    
    def __repr__(self) -> str:
        return (
            f"BNSimulator(name='{self.name}', "
            f"nodes={len(self.get_nodes())}, "
            f"edges={len(self.get_edges())})"
        )
