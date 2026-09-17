import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def parse_simulink_model_and_generate_descriptions(
    model_path: str, 
    signals: List, 
    simulation_time: float,
    short_version: bool = True,
    time_decimal_precision: int = 1
) -> List[str]:
    """
    Parse a Simulink model (.mdl or .slx) and generate dimension descriptions
    based on the model inputs and signal configuration.
    
    Args:
        model_path: Path to the .mdl or .slx file
        signals: List of SignalOptions objects containing:
                - control_points: Tuple of Interval objects for control point ranges
                - signal_times: List of time points (or None to generate automatically)
        simulation_time: Total simulation time
        short_version: If True, generate shorter descriptions without detailed context
        time_decimal_precision: Number of decimal places for time formatting (default: 1)
        
    Returns:
        List of dimension descriptions
    """
    
    # Extract model inputs
    inputs = extract_model_inputs(model_path)
    print(f"DEBUG: Extracted inputs from {model_path}: {inputs}")
    
    # Generate dimension descriptions
    dimension_descriptions = []
    
    if len(inputs) != len(signals):
        print(f"Warning: Number of inputs ({len(inputs)}) doesn't match signals ({len(signals)})")
        # Pad with generic inputs if needed
        while len(inputs) < len(signals):
            inputs.append({'name': f'input_{len(inputs)+1}', 'type': 'generic'})
    
    for i, (input_info, signal) in enumerate(zip(inputs, signals)):
        input_name = input_info.get('name', f'Input_{i+1}')
        print(f"DEBUG: Processing input {i}: {input_name}")
        
        # Get time points for this signal
        if signal.signal_times is not None:
            time_points = signal.signal_times
        else:
            # Generate time points if not provided
            num_points = len(signal.control_points)
            time_points = np.linspace(0, simulation_time, num_points)
        
        # Generate description for each control point
        for j, time_point in enumerate(time_points):
            description = generate_input_description(input_name, time_point, j, len(time_points), short_version=short_version, time_decimal_precision=time_decimal_precision)
            dimension_descriptions.append(description)
    
    return dimension_descriptions

def extract_model_inputs(model_path: str) -> List[Dict]:
    """
    Extract input information from Simulink model file.
    
    Args:
        model_path: Path to the .mdl or .slx file
        
    Returns:
        List of dictionaries containing input information
    """
    
    if model_path.endswith('.mdl'):
        return extract_inputs_from_mdl(model_path)
    elif model_path.endswith('.slx'):
        return extract_inputs_from_slx(model_path)
    else:
        raise ValueError("Unsupported file format. Only .mdl and .slx files are supported.")

def extract_inputs_from_mdl(mdl_path: str) -> List[Dict]:
    """Extract inputs from .mdl file"""
    inputs = []
    
    try:
        with Path(mdl_path).open('r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading .mdl file: {e}")
        return infer_inputs_from_model_name(mdl_path)
    
    # Look for ExternalInput configuration - handle different formats
    external_input_patterns = [
        r'ExternalInput\s+"([^"]+)"',  # Standard format: ExternalInput "[Throttle, Brake]"
        r'ExternalInput\s+\[([^\]]+)\]',  # Bracket format: ExternalInput [Throttle, Brake]
        r'ExternalInput\s+([^\s\n]+)',  # Simple format: ExternalInput Throttle
    ]
    
    for pattern in external_input_patterns:
        external_input_match = re.search(pattern, content)
        if external_input_match:
            input_names_str = external_input_match.group(1)
            # Clean up the string and extract names
            input_names_str = re.sub(r'[\[\]"]', '', input_names_str)  # Remove brackets and quotes
            input_names = [name.strip() for name in re.split(r'[,\s]+', input_names_str) if name.strip()]
            
            # Check if we got meaningful names or just placeholders
            generic_names = ['In1', 'In2', 'In3', 't__', 'u__', 'input1', 'input2', 'input3']
            meaningful_names = []
            
            for name in input_names:
                if name and name not in generic_names:
                    meaningful_names.append(name)
                    inputs.append({'name': name, 'type': 'external_input'})
            
            # If we only got generic names, don't use them - fall back to inference
            if meaningful_names:
                return inputs
            else:
                print(f"DEBUG: Found only generic names {input_names}, checking other sources")
                break
    
    # Look for Inport blocks - prioritize GraphicalInterface definitions (top-level inputs)
    graphical_interface_inputs = []
    
    # First, try to find GraphicalInterface Inport definitions (top-level inputs)
    # Use greedy match (.*) to handle nested braces by capturing until the end of the block (or file)
    # This is safe because Inport { ... } syntax is specific to GraphicalInterface
    graphical_interface_pattern = r'GraphicalInterface\s*\{(.*)\}'
    gi_match = re.search(graphical_interface_pattern, content, re.DOTALL)
    if gi_match:
        gi_content = gi_match.group(1)
        # Look for Inport blocks within GraphicalInterface
        gi_inport_pattern = r'Inport\s*\{[^}]*?Name\s*"([^"]+)"[^}]*?\}'
        gi_inport_matches = re.findall(gi_inport_pattern, gi_content, re.DOTALL)
        for match in gi_inport_matches:
            if match not in ['In1', 'In2', 'In3']:  # Skip generic names
                graphical_interface_inputs.append({'name': match, 'type': 'graphical_interface_inport'})
        
        if graphical_interface_inputs:
            print(f"DEBUG: Found GraphicalInterface inputs: {[inp['name'] for inp in graphical_interface_inputs]}")
            inputs.extend(graphical_interface_inputs)
    
    # If we don't have GraphicalInterface inputs or they're insufficient, look for regular Inport blocks
    if not graphical_interface_inputs:
        inport_patterns = [
            r'Block\s*\{\s*BlockType\s*"?Inport"?.*?Name\s*"([^"]+)".*?\}',
            r'BlockType\s*"?Inport"?[^}]*?Name\s*"([^"]+)"',
            r'Name\s*"([^"]+)"[^}]*BlockType\s*"?Inport"?',
        ]
        
        for pattern in inport_patterns:
            inport_matches = re.findall(pattern, content, re.DOTALL)
            for match in inport_matches:
                # Avoid duplicates and generic names
                if not any(inp['name'] == match for inp in inputs) and match not in ['In1', 'In2', 'In3']:
                    inputs.append({'name': match, 'type': 'inport'})
    
    # If no inputs found, try to infer from model name
    if not inputs:
        inputs = infer_inputs_from_model_name(mdl_path)
    
    return inputs

def extract_inputs_from_slx(slx_path: str) -> List[Dict]:
    """Extract inputs from .slx file"""
    inputs = []
    
    try:
        with zipfile.ZipFile(slx_path, 'r') as zip_file:
            # Read the block diagram XML
            try:
                with zip_file.open('simulink/blockdiagram.xml') as xml_file:
                    content = xml_file.read().decode('utf-8')
                    tree = ET.fromstring(content)
                    
                    # Look for Inport blocks in XML
                    for block in tree.findall('.//Block'):
                        if block.get('BlockType') == 'Inport':
                            # Look for name in P elements
                            for p_elem in block.findall('.//P'):
                                if p_elem.get('Name') == 'Name':
                                    name = p_elem.text.strip('"') if p_elem.text else None
                                    if name and name not in ['In1', 'In2', 'In3']:  # Skip generic names
                                        inputs.append({'name': name, 'type': 'inport'})
                                        break
                    
                    # Also look for System elements that might contain input names
                    for system in tree.findall('.//System'):
                        name_attr = system.get('Name')
                        if name_attr and 'input' in name_attr.lower():
                            inputs.append({'name': name_attr, 'type': 'system'})
                            
            except Exception as e:
                print(f"Error parsing blockdiagram.xml: {e}")
            
            # Also check graphical interface for input information
            try:
                with zip_file.open('simulink/graphicalInterface.xml') as xml_file:
                    content = xml_file.read().decode('utf-8')
                    tree = ET.fromstring(content)
                    
                    for inport in tree.findall('.//Inport'):
                        # Check for Name attribute first (common in newer files)
                        name = inport.get('Name')
                        
                        # Check for Name child element (older files)
                        if not name:
                            name_elem = inport.find('Name')
                            if name_elem is not None and name_elem.text:
                                name = name_elem.text
                        
                        if name:
                            name = name.strip()
                            # Avoid duplicates and generic names
                            if not any(inp['name'] == name for inp in inputs) and name not in ['In1', 'In2', 'In3']:
                                inputs.append({'name': name, 'type': 'inport'})
            except Exception as e:
                print(f"Error parsing graphicalInterface.xml: {e}")
                
    except Exception as e:
        print(f"Error reading .slx file: {e}")
        inputs = infer_inputs_from_model_name(slx_path)
    
    if not inputs:
        inputs = infer_inputs_from_model_name(slx_path)
    
    return inputs

def infer_inputs_from_model_name(model_path: str) -> List[Dict]:
    """Infer likely input names based on model filename"""
    model_name = os.path.splitext(os.path.basename(model_path))[0].lower()
    
    if 'autotrans' in model_name or 'car' in model_name:
        return [
            {'name': 'throttle', 'type': 'inferred'},
            {'name': 'brake', 'type': 'inferred'}
        ]
    elif 'narmamaglev' in model_name or 'nn' in model_name or 'steamcondense' in model_name or 'sc' in model_name:
        return [
            {'name': 'control_input', 'type': 'inferred'}
        ]
    else:
        # Default generic input
        return [
            {'name': 'input', 'type': 'inferred'}
        ]

def generate_input_description(input_name: str, time_point: float, index: int, total_points: int, short_version: bool = True, time_decimal_precision: int = 1) -> str:
    """Generate a descriptive name for an input at a specific time point
    
    Args:
        input_name: Name of the input signal
        time_point: Time point value
        index: Index of current point in timeline
        total_points: Total number of points in timeline
        short_version: If True, only return up to "t={time_point}s" without description after colon
        time_decimal_precision: Number of decimal places for time formatting (default: 1)
    """
    
    # Normalize input name for better descriptions
    input_name = input_name.lower()
    
    # Phase descriptions based on position in timeline
    if total_points == 1:
        phase = "control"
    elif index == 0:
        phase = "Initial"
    elif index == total_points - 1:
        phase = "Final"
    elif index / total_points < 0.25:
        phase = "Early phase"
    elif index / total_points < 0.5:
        phase = f"{'Mid-early' if total_points > 4 else 'Mid'} phase"
    elif index / total_points < 0.75:
        phase = f"{'Mid-late' if total_points > 4 else 'Mid'} phase"
    else:
        phase = "Late phase"
    
    # Input-specific descriptions
    if 'throttle' == input_name:
        base_desc = f"Throttle level at t={time_point:.{time_decimal_precision}f}s"
        if short_version:
            return base_desc
        action = "acceleration input" if phase == "control" else f"{phase.lower()} acceleration control"
        return f"{base_desc}: {action.capitalize()}"
    
    elif 'brake' == input_name:
        base_desc = f"Brake pressure at t={time_point:.{time_decimal_precision}f}s"
        if short_version:
            return base_desc
        action = "braking force" if phase == "control" else f"{phase.lower()} braking force"
        return f"{base_desc}: {action.capitalize()}"
    
    elif 'control' in input_name or 'input' in input_name or input_name in ['control_input', 'input']:
        base_desc = f"Control input at t={time_point:.{time_decimal_precision}f}s"
        if short_version:
            return base_desc
        return f"{base_desc}: {phase} control signal"
    
    elif input_name in ['fs', 'ref']:
        base_desc = f"Control input at t={time_point:.{time_decimal_precision}f}s"
        if short_version:
            return base_desc
        return f"{base_desc}: {phase} control signal"
    
    else:
        # Generic description
        base_desc = f"{input_name.capitalize()} at t={time_point:.{time_decimal_precision}f}s"
        if short_version:
            return base_desc
        return f"{base_desc}: {phase} {input_name} control"
