import typing
from tensor_annotations import axes


TEST_RUN = False

# file naming
CIS_FILE_START = "cis"
INSTANCE_VARS = "instancevars"
WEIGHTS = "weights"
SEQUENTIAL = "seq"
KWARGS = "kwargs"

# profiles for only loading/saving necessary data
FULL = "full"  # default for saving; loads/saves EVERYTHING including samples
TESTING = "testing"  # saves everything you need to test the performance
INFERENCE = "inference"  # saves only what is needed to generate p-values / CIs
# So, a net saved with FULL profile, can be loaded as testing or inference.
PROFILE_NESTING_ORDER = (INFERENCE, TESTING, FULL)

# model construction
NUM_HIDDEN_LAYERS = 7
NEURONS_PER_LAYER = 50

# layers
TANH_MULTIPLIER = 5.                                                           # we need to multiply the tanh up a bit if we want standard normal outputs
LEAKY_RELU_SLOPE = .3
NUM_RELU_IN_MULTIPLYER_LAYER = 5
LAYER_LEARNING_BATCH_SIZE = 1024
LAYER_LEARNING_RATE_INITIAL = .05
LAYER_LEARNING_HALF_LIFE_EPOCHS = .5
LAYER_LEARNING_STEPS_PER_EPOCH = 500
LAYER_LEARNING_EPOCHS = 4
LAYER_DEFAULT_IN = "linear"
LAYER_DEFAULT_HID = "multiplyer"
LAYER_DEFAULT_OUT = "linear"

# training
EPOCHS = 500
EPOCHS_ZNET = 1000
STEPS_PER_EPOCH = 100
STEPS_PER_EPOCH_ZNET = 200
BATCH_SIZE = 1024
SCHEDULE_FREE = False

# -- Adam:
AMS_GRAD_ADAM = False
LEARNING_RATE_INITIAL_ADAM = 0.0025
LEARNING_RATE_PLATEAU_PATIENCE_ADAM = 8
LEARNING_RATE_DECAY_RATIO_ON_PLATEAU_ADAM = 0.9
LEARNING_RATE_MINIMUM_ADAM = 1e-20

# -- Schedule-Free:
LEARNING_RATE_SCHEDULE_FREE = 0.005
LEARNING_WARMUP_STEPS_SCHEDULE_FREE = 1000

# training loss increase tolerances per net type
# TODO: Remove this and make it automated based on movement of losses
#       -- or at least find principled values!!
REL_LOSS_INCREASE_TOL_FEELER_NET = 10.
ABS_LOSS_INCREASE_TOL_PARAM_SAMP_NET = 8
ABS_LOSS_INCREASE_TOL_Z_NET = 2.
REL_LOSS_INCREASE_TOL_CI_NET = 1.3

# computation of ideal loss
MAX_PROPORTION_MISSING_VALUES_TO_TOLERATE = .1
NUM_SAMPLES_FOR_IDEAL_ERROR_ESTIMATION = 5000
GAP_BETWEEN_SAMPLES_FOR_PDF_ESTIMATION = 20

# Param sampling
FEELER_GENERATOR_BETA_DISTRIBUTION_BETA_AND_ALPHA = 1.                         # We want our beta distribution to be symmetric, so use the same value for alpha and beta.  A value below 1 will emphasise boundary points
FEELER_NET_MARKOV_CHAIN_LENGTH = 20000
FEELER_NET_NUM_CHAINS = 250
FEELER_NET_PERIPHERAL_BATCH_SIZE = 1250                                        # Set as high as your GPU can handle
FEELER_NET_PERIPHERAL_BATCHES = 800
OUTER_FEELER_PERIPHERAL_BATCH_SIZE = 1250
OUTER_FEELER_PERIPHERAL_BATCHES = 800
KNOWN_PARAM_MARKOV_CHAIN_SD = .005  # Will combine with step size
SAMPLES_PER_TEST_PARAM = 100
SAMPLE_PARAM_IF_SAMPLE_PERCENTILE = 99.9
IMPORTANCE_INGREDIENTS_VOLUMES_INDEX = 0
IMPORTANCE_INGREDIENTS_SHOULD_SAMPLE_INDEX = 1
IMPORTANCE_INGREDIENTS_VOLUMES_SMOOTHING = 0.0
IMPORTANCE_INGREDIENTS_SHOULD_SAMPLE_SMOOTHING = 0.0
# TODO: These will soon no longer apply, once params are properly sampled
PARAMS_MIN = -1.
PARAMS_MAX = 1.

# TODO: All of these to be chosen in a more principled way.
IS_INSIDE_NET_THRESHOLD = 0.01
INNER_FEELER_INCLUDE_THRESHOLD = -50.
OUTER_FEELER_INCLUDE_THRESHOLD = -75.
OUTER_FEELER_INCLUDE_BOOST = 3.

# other
DEFAULT_CONFIDENCE_LEVEL = .95
DZ0_DINTEREST_PENALTY_WEIGHT = 100.
ZNET_ANALYSER_NUM_SAMPLES = 1000
SAMPLES_TO_TEST_PARAM_MAPPINGS = 1024
ERROR_ALLOWED_FOR_PARAM_MAPPINGS = 1e-5
MIN_ALLOWED_JACOBDET_IN_COORDINET = -50.
SMALLEST_LOGABLE_NUMBER = 1e-37
NEGLIGIBLE_LOG = -5.
SOFT_FLOOR_CEILING = 1e-10
MAX_SAMPLES_AT_A_TIME = 50000


if TEST_RUN:
    print("\n\n\n\n           WARNING!!! \n\n\n This is a test run!!\n\n\n\n")
    EPOCHS = 10
    EPOCHS_ZNET = 13
    STEPS_PER_EPOCH = 2
    STEPS_PER_EPOCH_ZNET = 2

    FEELER_NET_MARKOV_CHAIN_LENGTH = 10
    FEELER_NET_PERIPHERAL_BATCH_SIZE = 500
    FEELER_NET_PERIPHERAL_BATCHES = 4
    OUTER_FEELER_PERIPHERAL_BATCH_SIZE = 125
    OUTER_FEELER_PERIPHERAL_BATCHES = 8


# axis labels for TensorFlow typing
# each of these presents what sort of data populates a certain axis in a
#  Tensor.
Batch = typing.NewType("Batch", axes.Axis)
Samples = typing.NewType("Samples", axes.Axis)
Params = typing.NewType("Params", axes.Axis)
UnknownParams = typing.NewType("UnknownParams", axes.Axis)
KnownParams = typing.NewType("KnownParams", axes.Axis)
FocalParam = typing.NewType("FocalParam", axes.Axis)
NuisanceParams = typing.NewType("NuisanceParams", axes.Axis)
Stats = typing.NewType("Stats", axes.Axis)
Us = typing.NewType("Us", axes.Axis)
Ys = typing.NewType("Ys", axes.Axis)
Zs = typing.NewType("Zs", axes.Axis)
NetOutputs = typing.NewType("NetOutputs", axes.Axis)
NetInputs = typing.NewType("NetInputs", axes.Axis)
LayerInputs = typing.NewType("LayerInputs", axes.Axis)
LayerOutputs = typing.NewType("LayerOutputs", axes.Axis)
NodesInLayer = typing.NewType("NodesInLayer", axes.Axis)
NumApproximations = typing.NewType("NumApproximations", axes.Axis)
MinAndMax = typing.NewType("MinAndMax", axes.Axis)
ImportanceIngredients = typing.NewType("ImportanceIngredients", axes.Axis)
TrainingBatches = typing.NewType("TrainingBatches", axes.Axis)
FixedParams = typing.NewType("FixedParams", axes.Axis)
ParamsAndKS = typing.NewType("ParamsAndKS", axes.Axis)
One = typing.NewType("One", axes.Axis)
Interest = typing.NewType("Interest", axes.Axis)
J = typing.NewType("J", axes.Axis)
Indices = typing.NewType("Indices", axes.Axis)
Chains = typing.NewType("Chains", axes.Axis)

# Again for typing, represents a blob of output from a network that can be
#   passed into a loss function.
NetInputBlob = typing.TypeVar("NetInputBlob")
NetOutputBlob = typing.TypeVar("NetOutputBlob")
NetTargetBlob = typing.TypeVar("NetTargetBlob")
NetInputSimulationBlob = typing.TypeVar("NetInputSimulationBlob")
