
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
 

SCDataT = NDArray[np.float_]
SCResultT = ExtraResult[SCDataT, SCDataT]


class SCModel(Model[SCResultT, None]):
    MODEL_NAME = "steamcondense_RNN_22"

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

    def simulate(self, inputs: ModelInputs, intrvl: Interval) -> SCResultT:
        sim_t = matlab.double([0, intrvl.upper])
        n_times = (intrvl.length // self.sampling_step) + 2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in inputs.signals])

        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())

        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )

        timestamps_array = np.array(timestamps).flatten()
        data_array = np.array(data)
        
        outTrace = Trace(timestamps_array, data_array)
        inTrace = Trace(signal_times, signal_values)
        return SCResultT(outTrace, inTrace)



if __name__ == "__main__":
    #####################################################################################################################
    # Define Specifications (all specifications)

    SCa_phi = "G[30,35] ((pressure <= 87.5) and (pressure >= 87))"

    spec_dict = {
        "SCa": RTAMTDense(SCa_phi, {"pressure":3}),
        }

    #####################################################################################################
    # Define Signals

    signals = [
        SignalOptions(control_points = [(3.99, 4.01)]*18, signal_times=np.linspace(0.,35.,18)),
    ]

    #####################################################################################################

    # Parse command line arguments
    parser = create_common_parser(
        "Run SC specification analysis",
        list(spec_dict.keys()),
        "SCa"
    )
    args = parser.parse_args()
    
    # Create model and runner
    model = SCModel()
    runner = SpecificationRunner(
        system_name="sc",
        model=model,
        spec_dict=spec_dict,
        signals=signals,
        interval=(0, 35)
    )
    
    # Define LLMGB-specific descriptions
    # Signal 1: 18 control points for steam condenser control input (e.g., valve position, heat input)
    signal_times = signals[0].signal_times
    dimension_descriptions_long = [
        f"Steam condenser control at t={t:.1f}s: Controls condensation rate"
        for t in signal_times
    ]

    if args.input_descriptions == 1:
        dimension_descriptions = model_inputs_extraction.parse_simulink_model_and_generate_descriptions("steamcondense_RNN_22.slx", signals, 35.0, short_version=True)
    elif args.input_descriptions == 2:
        dimension_descriptions = model_inputs_extraction.parse_simulink_model_and_generate_descriptions("steamcondense_RNN_22.slx", signals, 35.0, short_version=False)
    else:
        dimension_descriptions = dimension_descriptions_long
    
    output_descriptions = {
        "pressure": "System pressure in the steam condenser (units: likely PSI or bar)."
    }

    output_descriptions = output_descriptions if args.has_output_descriptions else None
    
    # Run analysis
    robustness, best_result = runner.run_analysis(args, dimension_descriptions, output_descriptions)