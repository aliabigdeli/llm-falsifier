"""
Critical Time Finder for STL Specifications

This module provides functionality to find critical time points in STL (Signal Temporal Logic)
specifications that determine the overall robustness value.
"""

import re
import rtamt
import numpy as np
from typing import Union, List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class OperatorType(Enum):
    """Types of STL operators"""
    ALWAYS = "always"
    EVENTUALLY = "eventually"
    AND = "and"
    OR = "or"
    IMPLIES = "implies"
    NOT = "not"
    ATOMIC = "atomic"


@dataclass
class STLNode:
    """Represents a node in the STL formula AST"""
    operator: OperatorType
    formula_str: str
    interval: Optional[Tuple[float, float]] = None
    children: List['STLNode'] = None
    variables: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.variables is None:
            self.variables = []


def fill_equal_robustness(
    total_time_points_list: Union[List[float], np.ndarray],
    robustness_time_list: Union[List[float], np.ndarray],
    robustness_values_list: Union[List[float], np.ndarray]
) -> np.ndarray:
    """
    Fill robustness values for all time points using forward-fill (last observation carried forward).
    
    Args:
        total_time_points_list: Complete list of time points to fill
        robustness_time_list: Time points where robustness values are known
        robustness_values_list: Known robustness values corresponding to robustness_time_list
    
    Returns:
        NumPy array of robustness values for all time points in total_time_points_list
    """
    total_times = np.asarray(total_time_points_list) if not isinstance(total_time_points_list, np.ndarray) else total_time_points_list
    rob_times = np.asarray(robustness_time_list) if not isinstance(robustness_time_list, np.ndarray) else robustness_time_list
    rob_values = np.asarray(robustness_values_list) if not isinstance(robustness_values_list, np.ndarray) else robustness_values_list
    
    indices = np.searchsorted(rob_times, total_times, side='right') - 1
    indices = np.clip(indices, 0, len(rob_values) - 1)
    
    if indices[0] < 0 or total_times[0] < rob_times[0]:
        raise ValueError(f"No known robustness value before time {total_times[0]}")
    
    filled_robustness = rob_values[indices]
    return filled_robustness


class STLFormulaParser:
    """Parser for STL formulas"""
    
    def __init__(self):
        self.patterns = {
            'temporal': r'(always|eventually|G|F)\s*\[([^\]]+)\]',
            'binary': r'\b(and|or)\b',
            'implies': r'->',
            'not': r'\bnot\b',
            'atomic': r'[a-zA-Z_][a-zA-Z0-9_]*\s*([><=]+)\s*[0-9.]+',
        }
    
    def parse(self, formula: str) -> STLNode:
        """Parse an STL formula string into an AST"""
        formula = formula.strip()
        
        # Remove outer parentheses if they wrap the entire formula (do this FIRST)
        if formula.startswith('(') and formula.endswith(')'):
            if self._matching_paren(formula, 0) == len(formula) - 1:
                return self.parse(formula[1:-1])
        
        # Handle parentheses - find top-level binary operators
        # Try to find implication first (lowest precedence)
        implies_pos = self._find_top_level_operator(formula, '->')
        if implies_pos >= 0:
            left = formula[:implies_pos].strip()
            right = formula[implies_pos+2:].strip()
            variables = self._extract_variables(formula)
            
            return STLNode(
                operator=OperatorType.IMPLIES,
                formula_str=formula,
                children=[self.parse(left), self.parse(right)],
                variables=variables
            )
        
        # Try to find OR
        or_pos = self._find_top_level_operator(formula, r'\bor\b')
        if or_pos >= 0:
            parts = self._split_by_operator(formula, r'\bor\b')
            variables = self._extract_variables(formula)
            
            return STLNode(
                operator=OperatorType.OR,
                formula_str=formula,
                children=[self.parse(p) for p in parts],
                variables=variables
            )
        
        # Try to find AND
        and_pos = self._find_top_level_operator(formula, r'\band\b')
        if and_pos >= 0:
            parts = self._split_by_operator(formula, r'\band\b')
            variables = self._extract_variables(formula)
            
            return STLNode(
                operator=OperatorType.AND,
                formula_str=formula,
                children=[self.parse(p) for p in parts],
                variables=variables
            )
        
        # Try to find NOT
        not_match = re.match(r'^\s*not\s*\((.+)\)\s*$', formula, re.IGNORECASE)
        if not_match:
            inner_formula = not_match.group(1)
            variables = self._extract_variables(formula)
            
            return STLNode(
                operator=OperatorType.NOT,
                formula_str=formula,
                children=[self.parse(inner_formula)],
                variables=variables
            )
        
        # Try to parse temporal operators (after checking binary operators)
        temporal_match = re.match(r'^\s*(always|eventually|G|F)\s*\[([^\]]+)\]\s*\((.+)\)\s*$', formula, re.IGNORECASE)
        if temporal_match:
            op_str = temporal_match.group(1).lower()
            interval_str = temporal_match.group(2)
            inner_formula = temporal_match.group(3)
            
            interval_parts = interval_str.split(',')
            interval = (float(interval_parts[0].strip()), float(interval_parts[1].strip()))
            
            operator = OperatorType.ALWAYS if op_str in ['always', 'g'] else OperatorType.EVENTUALLY
            child = self.parse(inner_formula)
            variables = self._extract_variables(formula)
            
            return STLNode(
                operator=operator,
                formula_str=formula,
                interval=interval,
                children=[child],
                variables=variables
            )
        
        # If nothing else matches, it's an atomic formula
        variables = self._extract_variables(formula)
        return STLNode(
            operator=OperatorType.ATOMIC,
            formula_str=formula,
            variables=variables
        )
    
    def _find_top_level_operator(self, formula: str, operator: str) -> int:
        """Find the position of a top-level operator (not inside parentheses)"""
        paren_depth = 0
        
        if operator.startswith('\\b') and operator.endswith('\\b'):
            # Word boundary operator (and, or)
            op_word = operator[2:-2]
            i = 0
            while i < len(formula):
                if formula[i] == '(':
                    paren_depth += 1
                elif formula[i] == ')':
                    paren_depth -= 1
                elif paren_depth == 0:
                    # Check if we have the operator word at this position
                    if formula[i:i+len(op_word)] == op_word:
                        # Check word boundaries
                        before_ok = (i == 0 or not formula[i-1].isalnum())
                        after_ok = (i + len(op_word) >= len(formula) or not formula[i+len(op_word)].isalnum())
                        if before_ok and after_ok:
                            return i
                i += 1
        else:
            # Simple operator (->)
            for i, char in enumerate(formula):
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif paren_depth == 0 and formula[i:i+len(operator)] == operator:
                    return i
        
        return -1
    
    def _split_by_operator(self, formula: str, operator: str) -> List[str]:
        """Split formula by top-level occurrences of an operator"""
        parts = []
        paren_depth = 0
        current_part = []
        
        if operator.startswith('\\b') and operator.endswith('\\b'):
            op_word = operator[2:-2]
            i = 0
            while i < len(formula):
                if formula[i] == '(':
                    paren_depth += 1
                    current_part.append(formula[i])
                    i += 1
                elif formula[i] == ')':
                    paren_depth -= 1
                    current_part.append(formula[i])
                    i += 1
                elif paren_depth == 0 and formula[i:i+len(op_word)] == op_word:
                    before_ok = (i == 0 or not formula[i-1].isalnum())
                    after_ok = (i + len(op_word) >= len(formula) or not formula[i+len(op_word)].isalnum())
                    if before_ok and after_ok:
                        parts.append(''.join(current_part).strip())
                        current_part = []
                        i += len(op_word)
                    else:
                        current_part.append(formula[i])
                        i += 1
                else:
                    current_part.append(formula[i])
                    i += 1
        else:
            for char in formula:
                if char == '(':
                    paren_depth += 1
                    current_part.append(char)
                elif char == ')':
                    paren_depth -= 1
                    current_part.append(char)
                elif paren_depth == 0 and ''.join(current_part[-len(operator)+1:] + [char]) == operator:
                    # Remove the operator chars we just added
                    for _ in range(len(operator) - 1):
                        current_part.pop()
                    parts.append(''.join(current_part).strip())
                    current_part = []
                else:
                    current_part.append(char)
        
        if current_part:
            parts.append(''.join(current_part).strip())
        
        return [p for p in parts if p]
    
    def _matching_paren(self, formula: str, start: int) -> int:
        """Find the matching closing parenthesis"""
        if formula[start] != '(':
            return -1
        
        depth = 1
        for i in range(start + 1, len(formula)):
            if formula[i] == '(':
                depth += 1
            elif formula[i] == ')':
                depth -= 1
                if depth == 0:
                    return i
        return -1
    
    def _extract_variables(self, formula: str) -> List[str]:
        """Extract variable names from formula"""
        # Find all identifiers that look like variable names
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        keywords = {'always', 'eventually', 'and', 'or', 'not', 'G', 'F'}
        
        variables = set()
        for match in re.finditer(pattern, formula):
            var = match.group(1)
            if var not in keywords:
                variables.add(var)
        
        return list(variables)


class CriticalTimeFinder:
    """Find critical time points in STL specifications"""
    
    def __init__(self, formula_str: str, signal_mapping: Dict[str, int], verbose: bool = False):
        """
        Initialize the critical time finder.
        
        Args:
            formula_str: STL formula as a string
            signal_mapping: Dictionary mapping signal names to trace column indices
            verbose: If True, print detailed information during search
        """
        self.formula_str = formula_str
        self.signal_mapping = signal_mapping
        self.verbose = verbose
        self.parser = STLFormulaParser()
        self.ast = self.parser.parse(formula_str)
        self.specs_cache = {}
    
    def _create_spec(self, formula: str, variables: List[str]) -> rtamt.StlDenseTimeSpecification:
        """Create an RTAMT specification for a formula"""
        if formula in self.specs_cache:
            return self.specs_cache[formula]
        
        spec = rtamt.StlDenseTimeSpecification()
        for var in variables:
            spec.declare_var(var, 'float')
        spec.spec = formula
        spec.parse()
        
        self.specs_cache[formula] = spec
        return spec
    
    def _evaluate_node(self, node: STLNode, trace_data: Dict[str, List]) -> np.ndarray:
        """Evaluate a node and return its robustness trace"""
        spec = self._create_spec(node.formula_str, node.variables)
        
        # Prepare trace arguments for evaluation
        trace_args = []
        for var in node.variables:
            if var in trace_data:
                time_points = trace_data['time']
                signal_values = trace_data[var]
                trace_list = [[time_points[i], signal_values[i]] for i in range(len(time_points))]
                trace_args.append((var, trace_list))
        
        # Evaluate
        if len(trace_args) == 0:
            raise ValueError(f"No trace data found for variables: {node.variables}")
        elif len(trace_args) == 1:
            rob = spec.evaluate(trace_args[0])
        else:
            rob = spec.evaluate(*trace_args)
        
        return rob
    
    def find_critical_time(self, trace_data: Dict[str, List]) -> Dict[str, Any]:
        """
        Find the critical time for the STL formula.
        
        Args:
            trace_data: Dictionary with 'time' key and signal keys mapping to lists of values
        
        Returns:
            Dictionary containing:
                - critical_time: The critical time point
                - critical_index: Index in the trace
                - robustness: Overall robustness value
                - path: List of formulas in the path to the critical time
                - details: Additional details about the critical path
        """
        time_points = np.array(trace_data['time'])
        
        # Evaluate the full formula
        overall_rob = self._evaluate_node(self.ast, trace_data)
        overall_robustness = overall_rob[0][1]
        
        if self.verbose:
            print(f"Overall robustness: {overall_robustness:.4f}")
            print(f"Analyzing formula: {self.formula_str}\n")
        
        # Find critical time recursively
        result = self._find_critical_time_recursive(
            self.ast,
            trace_data,
            time_points,
            start_index=0,
            end_index=len(time_points) - 1,
            path=[]
        )
        
        result['robustness'] = overall_robustness
        return result
    
    def _find_critical_time_recursive(
        self,
        node: STLNode,
        trace_data: Dict[str, List],
        time_points: np.ndarray,
        start_index: int,
        end_index: int,
        path: List[str]
    ) -> Dict[str, Any]:
        """Recursively find the critical time through the formula tree"""
        
        current_path = path + [node.formula_str]
        
        if self.verbose:
            print(f"Analyzing node: {node.operator.value}")
            print(f"  Formula: {node.formula_str}")
        
        # Base case: atomic formula
        if node.operator == OperatorType.ATOMIC:
            # For atomic formulas, the critical time is already determined by the parent
            # temporal operator (or it's the start_index if this is the root)
            critical_index = start_index
            return {
                'critical_time': time_points[critical_index],
                'critical_index': critical_index,
                'path': current_path,
                'details': f"Atomic formula: {node.formula_str}"
            }
        
        # Handle temporal operators
        if node.operator in [OperatorType.ALWAYS, OperatorType.EVENTUALLY]:
            interval_start, interval_end = node.interval
            
            # Find indices within the interval [start_index + interval_start, start_index + interval_end]
            effective_start_time = time_points[start_index] + interval_start
            effective_end_time = time_points[start_index] + interval_end
            
            valid_indices = np.where(
                (time_points >= effective_start_time) &
                (time_points <= effective_end_time)
            )[0]
            
            # Evaluate the CHILD formula (not the temporal operator itself)
            # to find which time point in the interval has min/max robustness
            child_rob_result = self._evaluate_node(node.children[0], trace_data)
            child_rob_times = np.array([t for t, v in child_rob_result])
            child_rob_values = np.array([v for t, v in child_rob_result])
            child_rob_values_filled = fill_equal_robustness(time_points, child_rob_times, child_rob_values)
            
            if len(valid_indices) == 0:
                # Use the start index if no valid indices
                critical_index = start_index
            else:
                if node.operator == OperatorType.ALWAYS:
                    # Find minimum robustness in the interval
                    local_critical = np.argmin(child_rob_values_filled[valid_indices])
                    critical_index = valid_indices[local_critical]
                else:  # EVENTUALLY
                    # Find maximum robustness in the interval
                    local_critical = np.argmax(child_rob_values_filled[valid_indices])
                    critical_index = valid_indices[local_critical]
            
            critical_time = time_points[critical_index]
            
            if self.verbose:
                print(f"  Interval: [{interval_start}, {interval_end}]")
                print(f"  Critical time: {critical_time:.2f} (index {critical_index})")
                print(f"  Child robustness at critical time: {child_rob_values_filled[critical_index]:.4f}\n")
            
            # Recurse into the child with updated interval
            # Calculate proper end_index based on actual time values
            interval_end_time = time_points[critical_index] + (interval_end - interval_start)
            child_end_index = np.searchsorted(time_points, interval_end_time, side='right')
            child_end_index = min(child_end_index, len(time_points) - 1, end_index)
            
            return self._find_critical_time_recursive(
                node.children[0],
                trace_data,
                time_points,
                critical_index,
                child_end_index,
                current_path
            )
        
        # Handle binary operators (AND, OR with multiple children)
        elif node.operator in [OperatorType.AND, OperatorType.OR]:
            # Evaluate ALL children at the current time point
            children_robustness = []
            
            for i, child in enumerate(node.children):
                child_rob = self._evaluate_node(child, trace_data)
                child_times = np.array([t for t, v in child_rob])
                child_values = np.array([v for t, v in child_rob])
                child_values_filled = fill_equal_robustness(time_points, child_times, child_values)
                rob_at_start = child_values_filled[start_index]
                children_robustness.append((i, rob_at_start, child))
                
                if self.verbose:
                    print(f"  Child {i+1} robustness at t={time_points[start_index]:.2f}: {rob_at_start:.4f}")
            
            if node.operator == OperatorType.AND:
                # For AND, the limiting branch is the one with MINIMUM robustness
                # (AND takes min of all children)
                limiting_idx, limiting_rob, limiting_child = min(children_robustness, key=lambda x: x[1])
                if self.verbose:
                    print(f"  Child {limiting_idx+1} is limiting (minimum robustness: {limiting_rob:.4f})\n")
                return self._find_critical_time_recursive(
                    limiting_child, trace_data, time_points, start_index, end_index, current_path
                )
            else:  # OR
                # For OR, the limiting branch is the one with MAXIMUM robustness
                # (OR takes max of all children)
                limiting_idx, limiting_rob, limiting_child = max(children_robustness, key=lambda x: x[1])
                if self.verbose:
                    print(f"  Child {limiting_idx+1} is limiting (maximum robustness for OR: {limiting_rob:.4f})\n")
                return self._find_critical_time_recursive(
                    limiting_child, trace_data, time_points, start_index, end_index, current_path
                )
        
        # Handle implication
        elif node.operator == OperatorType.IMPLIES:
            # (p -> q) ≡ (¬p ∨ q), robustness = max(-p, q)
            child1_rob = self._evaluate_node(node.children[0], trace_data)
            child1_times = np.array([t for t, v in child1_rob])
            child1_values = np.array([v for t, v in child1_rob])
            child1_values_filled = fill_equal_robustness(time_points, child1_times, child1_values)
            
            child2_rob = self._evaluate_node(node.children[1], trace_data)
            child2_times = np.array([t for t, v in child2_rob])
            child2_values = np.array([v for t, v in child2_rob])
            child2_values_filled = fill_equal_robustness(time_points, child2_times, child2_values)
            
            rob_neg_child1 = -child1_values_filled[start_index]
            rob_child2 = child2_values_filled[start_index]
            
            if self.verbose:
                print(f"  Antecedent (¬p) robustness: {rob_neg_child1:.4f}")
                print(f"  Consequent (q) robustness: {rob_child2:.4f}")
            
            # The limiting branch is the one with higher robustness
            if rob_neg_child1 >= rob_child2:
                if self.verbose:
                    print(f"  Antecedent (negated) is limiting\n")
                return self._find_critical_time_recursive(
                    node.children[0], trace_data, time_points, start_index, end_index, current_path
                )
            else:
                if self.verbose:
                    print(f"  Consequent is limiting\n")
                return self._find_critical_time_recursive(
                    node.children[1], trace_data, time_points, start_index, end_index, current_path
                )
        
        # Handle NOT
        elif node.operator == OperatorType.NOT:
            # For NOT, just recurse into the child
            return self._find_critical_time_recursive(
                node.children[0], trace_data, time_points, start_index, end_index, current_path
            )
        
        # Default case
        return {
            'critical_time': time_points[start_index],
            'critical_index': start_index,
            'path': current_path,
            'details': f"Unhandled operator: {node.operator}"
        }


def find_critical_time(
    formula: str,
    signal_mapping: Dict[str, int],
    trace_data: Dict[str, List],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Find the critical time for an STL formula given a trace.
    
    Args:
        formula: STL formula as a string (e.g., "G[0,20] (speed <= 120)")
        signal_mapping: Dictionary mapping signal names to trace column indices
        trace_data: Dictionary with 'time' key and signal keys mapping to lists of values
        verbose: If True, print detailed information during search
    
    Returns:
        Dictionary containing:
            - critical_time: The critical time point
            - critical_index: Index in the trace
            - robustness: Overall robustness value
            - path: List of formulas in the path to the critical time
            - details: Additional details about the critical path
    
    Example:
        >>> formula = "G[0,20] (speed <= 120)"
        >>> signal_mapping = {"speed": 0}
        >>> trace_data = {
        ...     'time': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ...     'speed': [100, 110, 115, 125, 120, 115]
        ... }
        >>> result = find_critical_time(formula, signal_mapping, trace_data)
        >>> print(f"Critical time: {result['critical_time']}")
        >>> print(f"Robustness: {result['robustness']}")
    """
    finder = CriticalTimeFinder(formula, signal_mapping, verbose=verbose)
    return finder.find_critical_time(trace_data)

