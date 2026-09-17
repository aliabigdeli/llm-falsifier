from collections import OrderedDict
from math import pi

import numpy as np
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from aerobench.run_f16_sim import run_f16_sim
from llm_falsifier import SpecificationRunner, create_common_parser
from numpy.typing import NDArray

from staliro.core.interval import Interval
from staliro.core.model import ExtraResult, Model, ModelInputs, Trace
from staliro.specifications import RTAMTDense

F16DataT = NDArray[np.float_]
F16ResultT = ExtraResult[F16DataT, F16DataT]


class F16Model(Model[F16ResultT, None]):
    def __init__(self, static_params_map) -> None:
        self.F16_PARAM_MAP = static_params_map

    def get_static_params(self):
        static_params = []
        for param, config in self.F16_PARAM_MAP.items():
            if config['enabled']:
                static_params.append(config['range'])
        return static_params

    def _compute_initial_conditions(self, X):
        conditions = []
        index = 0

        for param, config in self.F16_PARAM_MAP.items():
            if config['enabled']:
                conditions.append(X[index])
                index = index + 1
            else:
                conditions.append(config['default'])

        return conditions

    def simulate(self, inputs: ModelInputs, intrvl: Interval) -> F16ResultT:
        init_cond = self._compute_initial_conditions(inputs.static)
        
        step = 1 / 30
        autopilot = GcasAutopilot(init_mode="roll", stdout=False, gain_str="old")

        result = run_f16_sim(init_cond, intrvl.upper, autopilot, step, extended_states=True)
        trajectories = result["states"][:, 11:12].astype(np.float64)

        timestamps = np.array(result["times"], dtype=(np.float32))
        outTrace = Trace(timestamps, trajectories)
        print(inputs.static)
        inTrace = inputs.static
        return F16ResultT(outTrace, inTrace)


#####################################################################################################################

F16_PARAM_MAP = OrderedDict({
    'air_speed': {
        'enabled': False,
        'default': 540
    },
    'angle_of_attack': {
        'enabled': False,
        'default': np.deg2rad(2.1215)
    },
    'angle_of_sideslip': {
        'enabled': False,
        'default': 0
    },
    'roll': {
        'enabled': True,
        'default': None,
        'range': (pi / 4) + np.array((-pi / 20, pi / 30)),
    },
    'pitch': {
        'enabled': True,
        'default': None,
        'range': (-pi / 2) * 0.8 + np.array((0, pi / 20)),
    },
    'yaw': {
        'enabled': True,
        'default': None,
        'range': (-pi / 4) + np.array((-pi / 8, pi / 8)),
    },
    'roll_rate': {
        'enabled': False,
        'default': 0
    },
    'pitch_rate': {
        'enabled': False,
        'default': 0
    },
    'yaw_rate': {
        'enabled': False,
        'default': 0
    },
    'northward_displacement': {
        'enabled': False,
        'default': 0
    },
    'eastward_displacement': {
        'enabled': False,
        'default': 0
    },
    'altitude': {
        'enabled': False,
        'default': 2338.0
    },
    'engine_power_lag': {
        'enabled': False,
        'default': 9
    }
})

if __name__ == "__main__":
    #####################################################################################################
    # Define F16 Specifications

    # Basic altitude safety - original specification
    F16a_phi = "G[0,15](altitude > 0)"

    spec_dict = {
        "F16a": RTAMTDense(F16a_phi, {"altitude": 0})
    }

    #####################################################################################################

    # Parse command line arguments
    parser = create_common_parser(
        "Run F16 specification analysis",
        list(spec_dict.keys()),
        "F16a"
    )
    args = parser.parse_args()
    
    # Create model and runner
    model = F16Model(F16_PARAM_MAP)
    initial_conditions = model.get_static_params()
    
    runner = SpecificationRunner(
        system_name="f16",
        model=model,
        spec_dict=spec_dict,
        signals=[],
        interval=(0, 15),
        static_parameters=initial_conditions
    )
    
    # Define LLMGB-specific descriptions
    # 3 static parameters: roll (PHI), pitch (THETA), yaw (PSI) initial conditions in radians
    dimension_descriptions_long = [
        "PHI - Initial roll angle in radians: Controls aircraft banking angle at simulation start (positive = right wing down)",
        "THETA - Initial pitch angle in radians: Controls aircraft nose up/down attitude at simulation start (positive = nose up)", 
        "PSI - Initial yaw angle in radians: Controls aircraft heading direction at simulation start (positive = nose right)"
    ]

    if args.input_descriptions == 1:
        dimension_descriptions = ["PHI", "THETA", "PSI"]
    elif args.input_descriptions == 2:
        dimension_descriptions = dimension_descriptions_long
    else:
        dimension_descriptions = dimension_descriptions_long
    
    output_descriptions = {
        "altitude": "Aircraft altitude in feet: Current height above ground level during flight"
    }

    output_descriptions = output_descriptions if args.has_output_descriptions else None 
    
    # Run analysis
    robustness, best_result = runner.run_analysis(args, dimension_descriptions, output_descriptions)