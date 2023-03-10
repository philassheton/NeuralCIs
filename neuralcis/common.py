import tensorflow as tf
import numpy as np
import typing

from tensor_annotations import axes


# file naming
INSTANCE_VARS = "instancevars"
WEIGHTS = "weights"
FILE_PATH = "savedmodels"

# model construction
VALIDATION_SET_SIZE = 10000
NUM_HIDDEN_LAYERS = 3
NEURONS_PER_LAYER = 100

# training
MAX_EPOCHS = 100
BATCHES_PER_EPOCH = 20
MINIBATCHES_PER_BATCH = 100
MINIBATCH_SIZE = 32
LEARNING_RATE_INITIAL = 1e-3
DIVIDE_AFTER_FLATTENING_FOR = 10
TARGET_VALIDATION_LOSS_SD = 1.
TARGET_VALIDATION_LOSS_SD_P_NET = 1.
TARGET_VALIDATION_LOSS_SD_CI_NET = 1e-6

# computation of ideal loss
MAX_PROPORTION_MISSING_VALUES_TO_TOLERATE = .1
NUM_SAMPLES_FOR_IDEAL_ERROR_ESTIMATION = 5000
GAP_BETWEEN_SAMPLES_FOR_PDF_ESTIMATION = 20

# other
PARAMS_MIN = -1.
PARAMS_MAX = 1.
DEFAULT_CONFIDENCE_LEVEL = .95
ZNET_ANALYSER_NUM_SAMPLES = 1000
SAMPLES_TO_TEST_PARAM_MAPPINGS = 1000
ERROR_ALLOWED_FOR_PARAM_MAPPINGS = 1e-5


# axis labels for TensorFlow typing
# each of these presents what sort of data populates a certain axis in a
#  Tensor.
Samples = typing.NewType("Samples", axes.Axis)
Params = typing.NewType("Params", axes.Axis)
KnownParams = typing.NewType("KnownParams", axes.Axis)
Estimates = typing.NewType("Estimates", axes.Axis)
NetOutputs = typing.NewType("NetOutputs", axes.Axis)
NetInputs = typing.NewType("NetInputs", axes.Axis)
LayerInputs = typing.NewType("LayerInputs", axes.Axis)
LayerOutputs = typing.NewType("LayerOutputs", axes.Axis)
NodesInLayer = typing.NewType("NodesInLayer", axes.Axis)
NumApproximations = typing.NewType("NumApproximations", axes.Axis)
MinAndMax = typing.NewType("MinAndMax", axes.Axis)
TrainingBatches = typing.NewType("TrainingBatches", axes.Axis)
FixedParams = typing.NewType("FixedParams", axes.Axis)
ParamsAndKS = typing.NewType("ParamsAndKS", axes.Axis)

# Again for typing, represents a blob of output from a network that can be
#   passed into a loss function.
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
