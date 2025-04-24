import tensorflow as tf
import pickle
import os
from tensorflow.python.eager.def_function import Function as TFFunction        # type: ignore
from neuralcis.variables import Variable

# for typing
from typing import Tuple, Union, Callable, Sequence, Dict, Optional
from typing import Type, TypeVar
from tensor_annotations.tensorflow import Tensor1
from tensor_annotations.tensorflow import float32 as tf32
from neuralcis.common import Samples


T = TypeVar("T", bound="NeuralCIsKWArgs")

class _TFFn(tf.Module):
    def __init__(
            self,
            underlying_fn: Optional[TFFunction],
    ) -> None:

        super().__init__("TFFn")
        self.underlying_fn = underlying_fn

    @tf.function
    def __call__(self, *args, **kwargs):
        return self.underlying_fn(*args, **kwargs)

    @staticmethod
    def get(
            func: Union[None, Callable, TFFunction],
    ) -> "_TFFn":

        if func is None:
            return _TFFn(None)
        elif isinstance(func, TFFunction):
            return _TFFn(func)
        elif isinstance(func, _TFFn):
            return func
        elif _TFFn.is_reloaded_TFFn(func):
            return _TFFn(func.underlying_fn)
        else:
            return _TFFn(tf.function(func))

    @staticmethod
    def is_reloaded_TFFn(obj):
        cls = obj.__class__
        return (
                hasattr(obj, "underlying_fn") and
                cls.__name__ == "_UserObject" and
                cls.__module__.startswith("tensorflow.python.saved_model.load")
        )

    def save(self, foldername, filename):
        spec = [tf.TensorSpec([None], tf.float32)
                for _ in range(self.num_args())]
        concrete = self.underlying_fn.get_concrete_function(*spec)
        path = os.path.join(foldername, filename)
        tf.saved_model.save(self, path, signatures={'underlying_fn': concrete})

    @classmethod
    def load(cls, foldername, filename):
        path = os.path.join(foldername, filename)
        tffn: _TFFn = tf.saved_model.load(path)
        return tffn

    def arg_names(self):
        return self.underlying_fn.function_spec.arg_names

    def num_args(self):
        return len(self.arg_names())

    def is_none(self):
        return self.underlying_fn is None


class _NeuralCIsKWArgs():
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]]
            ],
            contrast_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Tensor1[tf32, Samples]
            ],
            unknown_param_names: Sequence[str],
            stat_names: Sequence[str],
            known_param_names: Sequence[str],
            transform_on_params_fn: Optional[Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]]
            ]],
            transform_on_stats_fn: Optional[Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]],
            ]],
            transform_on_params_param_names: Optional[Sequence[str]],
            network_setup_args: Optional[Dict],
            variable_defs: Dict[str, Variable],
    ) -> None:

        self.sampling_distribution_fn = _TFFn.get(sampling_distribution_fn)
        self.contrast_fn = _TFFn.get(contrast_fn)

        self.unknown_param_names = unknown_param_names
        self.stat_names = stat_names
        self.known_param_names = known_param_names

        self.transform_on_params_fn = _TFFn.get(transform_on_params_fn)
        self.transform_on_stats_fn = _TFFn.get(transform_on_stats_fn)
        self.transform_on_params_param_names = transform_on_params_param_names

        self.network_setup_args = network_setup_args
        self.variable_defs = variable_defs

    def kwargs(self) -> Dict:
        return self._wrap_up_kwargs(
            sampling_distribution_fn = self.sampling_distribution_fn,
            contrast_fn = self.contrast_fn,
            unknown_param_names = self.unknown_param_names,
            stat_names = self.stat_names,
            known_param_names = self.known_param_names,
            transform_on_params_fn = self.transform_on_params_fn,
            transform_on_stats_fn = self.transform_on_stats_fn,
            transform_on_params_param_names =
                                        self.transform_on_params_param_names,
            network_setup_args = self.network_setup_args,
        ) | self.variable_defs

    @staticmethod
    def _wrap_up_kwargs(**kwargs):
        return kwargs

    def save(self, foldername) -> None:
        # Save tf_fn separately as need to use tf saving stuff for this
        self.sampling_distribution_fn.save(
            foldername, "sampling_distribution_fn",
        )
        self.contrast_fn.save(
            foldername, "contrast_fn",
        )
        self.transform_on_params_fn.save(
            foldername, "transform_on_params_fn",
        )
        self.transform_on_stats_fn.save(
            foldername, "transform_on_stats_fn",
        )

        # Pickle variable defs separately: need special treatment on loading
        self._save_pickle(foldername, "variable_defs", self.variable_defs)

        other_args = {"unknown_param_names": self.unknown_param_names,
                      "stat_names": self.stat_names,
                      "known_param_names": self.known_param_names,
                      "transform_on_params_param_names":
                                        self.transform_on_params_param_names,
                      "network_setup_args": self.network_setup_args}
        self._save_pickle(foldername, "other_args", other_args)

    @classmethod
    def load(
            cls: Type[T],
            foldername: str,
    ) -> T:
        kwargs = cls._wrap_up_kwargs(
            sampling_distribution_fn = _TFFn.load(
                foldername, "sampling_distribution_fn",
            ),
            contrast_fn = _TFFn.load(
                foldername, "contrast_fn",
            ),
            transform_on_params_fn = _TFFn.load(
                foldername, "transform_on_params_fn"
            ),
            transform_on_stats_fn = _TFFn.load(
                foldername, "transform_on_stats_fn",
            ),
        )

        other_args: Dict
        variable_defs: Dict[str, Variable]
        other_args = cls._load_pickle(foldername, "other_args")
        variable_defs = cls._load_pickle(foldername, "variable_defs")
        for variable_def in variable_defs.values():
            variable_def.reapply_tf_functions()

        kwargs |= {"variable_defs": variable_defs} | other_args
        return cls(**kwargs)

    @staticmethod
    def _load_pickle(
            foldername: str,
            filename: str,
    ) -> object:

        path = os.path.join(foldername, filename)
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _save_pickle(
            foldername: str,
            filename: str,
            thing_to_save: object
    ) -> None:

        path = os.path.join(foldername, filename)
        with open(path, 'wb') as f:
            pickle.dump(thing_to_save, f)
