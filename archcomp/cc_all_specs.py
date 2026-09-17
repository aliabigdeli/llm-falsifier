
import numpy as np
from llm_falsifier import SpecificationRunner, create_common_parser, model_inputs_extraction
from numpy.typing import NDArray

from staliro.core.interval import Interval
from staliro.core.model import ExtraResult, Model, ModelInputs, Trace
from staliro.options import SignalOptions
from staliro.specifications import RTAMTDense
from staliro.signals import piecewise_constant

try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True


CCDataT = NDArray[np.float_]
CCResultT = ExtraResult[CCDataT, CCDataT]


class CCModel(Model[CCResultT, None]):
    MODEL_NAME = "cars"

    def __init__(self) -> None:
        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )

        engine = matlab.engine.start_matlab()
        model_opts = engine.simget(self.MODEL_NAME)

        self.sampling_step = 0.05
        self.engine = engine
        self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")

    def simulate(self, inputs: ModelInputs, intrvl: Interval) -> CCResultT:
        sim_t = matlab.double([0, intrvl.upper])
        n_times = (intrvl.length // self.sampling_step) + 2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in inputs.signals])
        
        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())

        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )

        data_array = np.array(data)
        y54 = (data_array[:,4]-data_array[:,3]).reshape((-1,1))
        y43 = (data_array[:,3]-data_array[:,2]).reshape((-1,1))
        y32 = (data_array[:,2]-data_array[:,1]).reshape((-1,1))
        y21 = (data_array[:,1]-data_array[:,0]).reshape((-1,1))
        diff_array = np.hstack((y21, y32, y43, y54))
        timestamps_list = np.array(timestamps).flatten()
        data_list = np.array(diff_array)
        trace = Trace(timestamps_list, data_list)

        inTrace = Trace(signal_times, signal_values)
        return CCResultT(trace, inTrace)



if __name__ == "__main__":
    #####################################################################################################################
    # Define Specifications (all specifications)

    CC1_phi = "G[0, 100] (y54 <= 40)"
    CC2_phi = "G[0, 70] (F[0,30] (y54 >= 15))"
    CC3_phi = "G[0, 80] ((G[0, 20] (y21 <= 20)) or (F[0,20] (y54 >= 40)))"
    CC4_phi = "G[0,65] (F[0,30] (G[0,5] (y54 >= 8)))"
    CC5_phi = "G[0,72] (F[0,8] ((G[0,5] (y21 >= 9)) -> (G[5,20] (y54 >= 9))))"

    phi_1 = "(G[0, 50] (y21 >= 7.5))"
    phi_2 = "(G[0, 50] (y32 >= 7.5))"
    phi_3 = "(G[0, 50] (y43 >= 7.5))"
    phi_4 = "(G[0, 50] (y54 >= 7.5))"
    CCx_phi = phi_1 + " and " + phi_2 + " and " + phi_3 + " and " + phi_4

    spec_dict = {
        "CC1": RTAMTDense(CC1_phi, {"y54": 3}),
        "CC2": RTAMTDense(CC2_phi, {"y54": 3}),
        "CC3": RTAMTDense(CC3_phi, {"y21": 0, "y54":3}),
        "CC4": RTAMTDense(CC4_phi, {"y54": 3}),    
        "CC5": RTAMTDense(CC5_phi, {"y21": 0, "y54":3}),    
        "CCx": RTAMTDense(CCx_phi,{"y21":0, "y32":1, "y43":2, "y54":3}),
        }

    #####################################################################################################
    # Define Signals
    signals = [
        SignalOptions(control_points = [(0., 1.)] * 10, signal_times=np.linspace(0.0, 100.0, 10)),
        SignalOptions(control_points = [(0., 1.)] * 10, signal_times=np.linspace(0.0, 100.0, 10))
    ]

    #####################################################################################################
    
    
    # Parse command line arguments
    parser = create_common_parser(
        "Run CC specification analysis",
        list(spec_dict.keys()),
        "CC1"
    )
    args = parser.parse_args()
    
    # Create model and runner
    model = CCModel()
    runner = SpecificationRunner(
        system_name="cc",
        model=model,
        spec_dict=spec_dict,
        signals=signals,
        interval=(0, 100)
    )
    
    # Define LLMGB-specific descriptions
    dimension_descriptions_long = [
        # Signal 1: Throttle control for Car 1 (dimensions 0-9)
        "Car 1 throttle at t=0.0s: Initial throttle input for lead car",
        "Car 1 throttle at t=11.11s: Early phase throttle control for lead car", 
        "Car 1 throttle at t=22.22s: Early-mid phase throttle control for lead car",
        "Car 1 throttle at t=33.33s: Mid phase throttle control for lead car",
        "Car 1 throttle at t=44.44s: Mid-late phase throttle control for lead car",
        "Car 1 throttle at t=55.56s: Late phase throttle control for lead car",
        "Car 1 throttle at t=66.67s: Very late phase throttle control for lead car",
        "Car 1 throttle at t=77.78s: Near-final phase throttle control for lead car",
        "Car 1 throttle at t=88.89s: Pre-final phase throttle control for lead car",
        "Car 1 throttle at t=100.0s: Final throttle input for lead car",
        
        # Signal 2: Brake control for Car 1 (dimensions 10-19)  
        "Car 1 brake at t=0.0s: Initial brake input for lead car",
        "Car 1 brake at t=11.11s: Early phase brake control for lead car",
        "Car 1 brake at t=22.22s: Early-mid phase brake control for lead car",
        "Car 1 brake at t=33.33s: Mid phase brake control for lead car",
        "Car 1 brake at t=44.44s: Mid-late phase brake control for lead car",
        "Car 1 brake at t=55.56s: Late phase brake control for lead car", 
        "Car 1 brake at t=66.67s: Very late phase brake control for lead car",
        "Car 1 brake at t=77.78s: Near-final phase brake control for lead car",
        "Car 1 brake at t=88.89s: Pre-final phase brake control for lead car",
        "Car 1 brake at t=100.0s: Final brake input for lead car"
    ]

    if args.input_descriptions == 1:
        dimension_descriptions = model_inputs_extraction.parse_simulink_model_and_generate_descriptions("cars.mdl", signals, 100.0, short_version=True)
    elif args.input_descriptions == 2:
        dimension_descriptions = model_inputs_extraction.parse_simulink_model_and_generate_descriptions("cars.mdl", signals, 100.0, short_version=False)
    else:
        dimension_descriptions = dimension_descriptions_long
    
    output_descriptions = {
        "y21": "Distance between car 2 and car 1 (following distance)",
        "y32": "Distance between car 3 and car 2 (second following distance)",
        "y43": "Distance between car 4 and car 3 (third following distance)",
        "y54": "Distance between car 5 and car 4 (fourth following distance)"
    }

    output_descriptions = output_descriptions if args.has_output_descriptions else None 
    
    # Run analysis
    robustness, best_result = runner.run_analysis(args, dimension_descriptions, output_descriptions)