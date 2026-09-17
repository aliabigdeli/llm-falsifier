
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


AutotransDataT = NDArray[np.float_]
AutotransResultT = ExtraResult[AutotransDataT, AutotransDataT]
 
 
class AutotransModel(Model[AutotransDataT, None]):
    MODEL_NAME = "Autotrans_shift"
 
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
        print("Model Initialized")

    def simulate(self, signals: ModelInputs, intrvl: Interval) -> AutotransResultT:
        sim_t = matlab.double([0, intrvl.upper])
        n_times = (intrvl.length // self.sampling_step) + 2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in signals.signals])
 
        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())
        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )
 
        timestamps_list = np.array(timestamps).flatten()
        data_list = np.array(data)
        trace = Trace(timestamps_list, data_list)

        inTrace = Trace(signal_times, signal_values)
        return AutotransResultT(trace, inTrace)



if __name__ == "__main__":
    #####################################################################################################################
    # Define Specifications (all specifications)

    AT1_phi = "G[0, 20] (speed <= 120)"

    AT2_phi = "G[0, 10] (rpm <= 4750)"

    gear_1_phi = f"(gear <= 1.5 and gear >= 0.5)"
    AT51_phi = f"G[0, 30] (((not {gear_1_phi}) and (F[0.001,0.1] {gear_1_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_1_phi})))"

    gear_2_phi = f"(gear <= 2.5 and gear >= 1.5)"
    AT52_phi = f"G[0, 30] (((not {gear_2_phi}) and (F[0.001,0.1] {gear_2_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_2_phi})))"

    gear_3_phi = f"(gear <= 3.5 and gear >= 2.5)"
    AT53_phi = f"G[0, 30] (((not {gear_3_phi}) and (F[0.001,0.1] {gear_3_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_3_phi})))"

    gear_4_phi = f"(gear <= 4.5 and gear >= 3.5)"
    AT54_phi = f"G[0, 30] (((not {gear_4_phi}) and (F[0.001,0.1] {gear_4_phi})) -> (F[0.001, 0.1] (G[0,2.5] {gear_4_phi})))"

    AT6a_phi = "((G[0, 30] (rpm <= 3000)) -> (G[0,4] (speed <= 35)))"
    AT6b_phi = "((G[0, 30] (rpm <= 3000)) -> (G[0,8] (speed <= 50)))"
    AT6c_phi = "((G[0, 30] (rpm <= 3000)) -> (G[0,20] (speed <= 65)))"
    AT6abc_phi = f"{AT6a_phi} and {AT6b_phi} and {AT6c_phi}"

    spec_dict = {
        "AT1": RTAMTDense(AT1_phi, {"speed": 0}),
        "AT2": RTAMTDense(AT2_phi, {"rpm": 1}),
        "AT51": RTAMTDense(AT51_phi, {"gear": 2}),    
        "AT52": RTAMTDense(AT52_phi, {"gear": 2}),    
        "AT53": RTAMTDense(AT53_phi, {"gear": 2}),    
        "AT54": RTAMTDense(AT54_phi, {"gear": 2}),    
        "AT6a": RTAMTDense(AT6a_phi, {"speed": 0, "rpm":1}),
        "AT6b": RTAMTDense(AT6b_phi, {"speed": 0, "rpm":1}),
        "AT6c": RTAMTDense(AT6c_phi, {"speed": 0, "rpm":1}),
        "AT6abc": RTAMTDense(AT6abc_phi, {"speed": 0, "rpm":1}),
        }

    #####################################################################################################
    # Define Signals
    signals = [
        SignalOptions(control_points = [(0, 100)]*7, signal_times=np.linspace(0.,50.,7)),
        SignalOptions(control_points = [(0, 325)]*3, signal_times=np.linspace(0.,50.,3)),
    ]

    #####################################################################################################
    
    # Parse command line arguments
    parser = create_common_parser(
        "Run Autotrans specification analysis",
        list(spec_dict.keys()),
        "AT1"
    )
    args = parser.parse_args()
    
    # Create model and runner
    model = AutotransModel()
    runner = SpecificationRunner(
        system_name="autotrans",
        model=model,
        spec_dict=spec_dict,
        signals=signals,
        interval=(0, 50)
    )
    
    # Define LLMGB-specific descriptions
    dimension_descriptions_long = [
        "Throttle level at t=0.0s: Initial acceleration input",
        "Throttle level at t=8.33s: Early phase acceleration control", 
        "Throttle level at t=16.67s: Mid-early phase acceleration control",
        "Throttle level at t=25.0s: Mid phase acceleration control",
        "Throttle level at t=33.33s: Mid-late phase acceleration control",
        "Throttle level at t=41.67s: Late phase acceleration control",
        "Throttle level at t=50.0s: Final acceleration input",
        "Brake pressure at t=0.0s: Initial braking force",
        "Brake pressure at t=25.0s: Mid-simulation braking force", 
        "Brake pressure at t=50.0s: Final braking force"
    ]

    if args.input_descriptions == 1:
        dimension_descriptions = model_inputs_extraction.parse_simulink_model_and_generate_descriptions("Autotrans_shift.mdl", signals, 50.0, short_version=True)
    elif args.input_descriptions == 2:
        dimension_descriptions = model_inputs_extraction.parse_simulink_model_and_generate_descriptions("Autotrans_shift.mdl", signals, 50.0, short_version=False)
    else:
        dimension_descriptions = dimension_descriptions_long
    
    output_descriptions = {
        "speed": "Vehicle speed in mph - how fast the car is traveling",
        "rpm": "Engine RPM (revolutions per minute) - engine rotational speed", 
        "gear": "Current transmission gear (1=first, 2=second, 3=third, 4=fourth gear)"
    }

    output_descriptions = output_descriptions if args.has_output_descriptions else None

    # Run analysis
    robustness, best_result = runner.run_analysis(args, dimension_descriptions, output_descriptions=output_descriptions)