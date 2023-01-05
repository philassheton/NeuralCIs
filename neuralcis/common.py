import tensorflow as tf
import numpy as np
import typing

from tensor_annotations import axes

VALIDATION_SET_SIZE = 10000

# axis labels for TensorFlow typing
# each of these presents what sort of data populates a certain axis in a
#  Tensor.
Samples = typing.NewType("Samples", axes.Axis)
Params = typing.NewType("Params", axes.Axis)
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
