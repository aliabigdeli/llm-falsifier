
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
 

NNDataT = NDArray[np.float_]
NNResultT = ExtraResult[NNDataT, NNDataT]


class NNModel(Model[NNResultT, None]):
    MODEL_NAME = "narmamaglev_v1"

    def __init__(self) -> None:
        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )

        engine = matlab.engine.start_matlab()
        model_opts = engine.simget(self.MODEL_NAME)
        
        self.sampling_step = 0.01
        self.engine = engine
        self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")

        self.alpha_1 = 0.005
        self.beta_1 = 0.03
        
        self.alpha_2 = 0.005
        self.beta_2 = 0.04

    def simulate(self, inputs: ModelInputs, intrvl: Interval) -> NNResultT:
        sim_t = matlab.double([0, intrvl.upper])
        n_times = intrvl.length // self.sampling_step + 2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in inputs.signals])
        
        self.engine.workspace["u_ts"] = 0.001
        model_input = matlab.double(
            np.row_stack((signal_times, signal_values)).T.tolist()
        )
        
        timestamps, _, data = self.engine.sim(
            self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=3
        )
        
        timestamps_array = np.array(timestamps).flatten()
        data_array = np.array(data)
        
        pos = data_array[:,0]

        pos_ref_mod = np.absolute(data_array[:,0] - data_array[:,2])
        mod_ref = np.absolute(data_array[:,2])

        p1 = pos_ref_mod - (self.alpha_1 + (self.beta_1 * mod_ref))
        p2 = (self.alpha_1 + (self.beta_1 * mod_ref)) - pos_ref_mod

        p3 = pos_ref_mod - (self.alpha_2 + (self.beta_2 * mod_ref))
        p4 = (self.alpha_2 + (self.beta_2 * mod_ref)) - pos_ref_mod

        sim_data = np.vstack([pos, data_array[:,2], p1, p2, p3, p4])
        outTrace = Trace(timestamps_array, sim_data.T)
        inTrace = Trace(signal_times, signal_values)
        return NNResultT(outTrace, inTrace)



if __name__ == "__main__":
    #####################################################################################################################
    # Define Specifications (all specifications)

    p11_phi = "(p1 >= 0)"
    p12_phi = "(F[0,2] (G[0,1] (not(p2 <= 0))))"
    NN1_phi = f"G[1,37] ({p11_phi} -> {p12_phi})"

    p21_phi = "(p3 >= 0)"
    p22_phi = "(F[0,2] (G[0,1] (not(p4 <= 0))))"
    NN2_phi = f"G[1,37] ({p21_phi} -> {p22_phi})"

    phi_1 = "F[0,1] (pos >= 3.2)"
    phi_2 = "F[1,1.5] (G[0,0.5]((pos >= 1.75) and (pos <= 2.25)))"
    phi_3 = "G[2,3] ((pos >= 1.825) and (pos <= 2.175))"
    NNx_phi = f"{phi_1} and {phi_2} and {phi_3}"

    spec_dict = {
        "NN1": RTAMTDense(NN1_phi, {"p1": 2, "p2": 3}),
        "NN2": RTAMTDense(NN2_phi, {"p3": 4, "p4": 5}),
        "NNx": RTAMTDense(NNx_phi, {"pos": 0}),
        }

    #####################################################################################################
    # Define Signals
    signals_1_2 = [
        SignalOptions(control_points = [(1,3)]*8, signal_times=np.linspace(0.,40.,8)),
    ]

    signals_x = [
        SignalOptions(control_points = [(1.95,2.05)]*8, signal_times=np.linspace(0.,40.,8)),
    ]


    #####################################################################################################
    
    
    # Parse command line arguments
    parser = create_common_parser(
        "Run NN specification analysis",
        list(spec_dict.keys()),
        "NN1"
    )
    args = parser.parse_args()

    if args.spec == "NN1" or args.spec == "NN2":
        signals = signals_1_2
        interval = (0, 40)
    elif args.spec == "NNx":
        signals = signals_x
        interval = (0, 3)
    else:
        raise ValueError(f"Invalid specification: {args.spec}")
    
    # Create model and runner
    model = NNModel()
    runner = SpecificationRunner(
        system_name="nn",
        model=model,
        spec_dict=spec_dict,
        signals=signals,
        interval=interval
    )
    
    # Define LLMGB-specific descriptions
    # Signal 1: 8 control points for NN control input
    signal_times = signals[0].signal_times
    dimension_descriptions_long = []
    for i, time in enumerate(signal_times, 1):
        dimension_descriptions_long.append(
            f"Neural network control input at t={time:.1f}s: Controls the input to the neural network"
        )

    if args.input_descriptions == 1:
        dimension_descriptions = model_inputs_extraction.parse_simulink_model_and_generate_descriptions("narmamaglev_v1.slx", signals, interval[-1], short_version=True)
    elif args.input_descriptions == 2:
        dimension_descriptions = model_inputs_extraction.parse_simulink_model_and_generate_descriptions("narmamaglev_v1.slx", signals, interval[-1], short_version=False)
    else:
        dimension_descriptions = dimension_descriptions_long
    
    output_descriptions = {
        "pos": "Position of the magnetic levitation system (column 0)",
        "reference": "Reference position/setpoint for the maglev system (column 1)", 
        "p1": "Performance metric 1: tracking error minus tolerance bound (α₁=0.005, β₁=0.03) (column 2)",
        "p2": "Performance metric 2: tolerance bound minus tracking error (complement of p1) (column 3)",
        "p3": "Performance metric 3: tracking error minus tolerance bound (α₂=0.005, β₂=0.04) (column 4)",
        "p4": "Performance metric 4: tolerance bound minus tracking error (complement of p3) (column 5)"
    }
    

    output_descriptions = output_descriptions if args.has_output_descriptions else None
    
    # Run analysis
    robustness, best_result = runner.run_analysis(args, dimension_descriptions, output_descriptions)