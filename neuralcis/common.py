import tensorflow as tf
import numpy as np
import typing

from tensor_annotations import axes


# file naming
CIS_FILE_START = "cis"
INSTANCE_VARS = "instancevars"
WEIGHTS = "weights"
SEQUENTIAL = "seq"

# model construction
VALIDATION_SET_SIZE = 10000
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
STEPS_PER_EPOCH = 100
MINIBATCH_SIZE = 1024
LEARNING_RATE_INITIAL = 0.0025
AMS_GRAD = False

LEARNING_RATE_PLATEAU_PATIENCE = 8
LEARNING_RATE_DECAY_RATIO_ON_PLATEAU = 0.9
LEARNING_RATE_MINIMUM = 1e-6

# training loss increase tolerances per net type
# TODO: Remove this and make it automated based on movement of losses
ABS_LOSS_INCREASE_TOL_FEELER_NET = 0.
ABS_LOSS_INCREASE_TOL_PARAM_SAMP_NET = .5
ABS_LOSS_INCREASE_TOL_Z_NET = .3
REL_LOSS_INCREASE_TOL_CI_NET = 1.3

# computation of ideal loss
MAX_PROPORTION_MISSING_VALUES_TO_TOLERATE = .1
NUM_SAMPLES_FOR_IDEAL_ERROR_ESTIMATION = 5000
GAP_BETWEEN_SAMPLES_FOR_PDF_ESTIMATION = 20

# Param sampling
FEELER_NET_MARKOV_CHAIN_LENGTH = 1000
FEELER_NET_NUM_CHAINS = 100
FEELER_NET_NUM_PERIPHERAL_POINTS = 50000
PARAM_MARKOV_CHAIN_STEP_SIZE = 1.
KNOWN_PARAM_MARKOV_CHAIN_SD = .1  # Will combine with step size
SAMPLES_PER_TEST_PARAM = 100
SAMPLE_PARAM_IF_SAMPLE_PERCENTILE = 99.
# TODO: These will soon no longer apply, once params are properly sampled
PARAMS_MIN = -1.
PARAMS_MAX = 1.
# Feeler net validation set
FEELER_NET_VALIDATION_SET_MARKOV_CHAIN_LENGTH = 1000
FEELER_NET_VALIDATION_SET_NUM_CHAINS = 5
FELLER_NET_VALIDATION_SET_NUM_PERIPHERAL_SAMPLES = 5000



# other
DEFAULT_CONFIDENCE_LEVEL = .95
ZNET_ANALYSER_NUM_SAMPLES = 1000
SAMPLES_TO_TEST_PARAM_MAPPINGS = 1000
ERROR_ALLOWED_FOR_PARAM_MAPPINGS = 1e-5
MIN_ALLOWED_JACOBDET_IN_COORDINET = -50.
SMALLEST_LOGABLE_NUMBER = 1e-37
SOFT_FLOOR_CEILING = 1e-10


# axis labels for TensorFlow typing
# each of these presents what sort of data populates a certain axis in a
#  Tensor.
Samples = typing.NewType("Samples", axes.Axis)
Params = typing.NewType("Params", axes.Axis)
UnknownParams = typing.NewType("UnknownParams", axes.Axis)
KnownParams = typing.NewType("KnownParams", axes.Axis)
FocalParam = typing.NewType("FocalParam", axes.Axis)
NuisanceParams = typing.NewType("NuisanceParams", axes.Axis)
Estimates = typing.NewType("Estimates", axes.Axis)
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
Contrast = typing.NewType("Contrast", axes.Axis)
J = typing.NewType("J", axes.Axis)
Indices = typing.NewType("Indices", axes.Axis)

# Again for typing, represents a blob of output from a network that can be
#   passed into a loss function.
NetInputBlob = typing.TypeVar("NetInputBlob")
NetOutputBlob = typing.TypeVar("NetOutputBlob")
NetTargetBlob = typing.TypeVar("NetTargetBlob")


def combine_input_args_into_tensor(*argv):
    """Combine various different input formats into a TensorFlow Tensor.

    :param argv: Inputs to the network may be passed in here as any of:
    (1) A Tensor
    (2) A series of scalar values (one for each network input)
    (3) A series of 1D Tensors (one for each network input)
    :return: A Tensorflow Tensor.
    """
    if len(argv) == 1:
        argv = argv[0]

    if tf.is_tensor(argv):
        tensor = argv
    elif np.all([np.isscalar(arg) for arg in argv]):
        tensor = tf.stack([[arg] for arg in argv], axis=1)
    else:
        tensor = tf.stack(argv, axis=1)

    return tensor
