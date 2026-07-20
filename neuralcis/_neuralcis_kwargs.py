import tensorflow as tf
import pickle
import os
from datetime import datetime
from tensorflow.python.eager.def_function import Function as TFFunction        # type: ignore
from .variables import Variable

# for typing
from typing import Tuple, Union, Callable, Sequence, Dict, Optional
from typing import Type, TypeVar
from tensor_annotations.tensorflow import Tensor1
from tensor_annotations.tensorflow import float32 as tf32
from .common import Samples


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
            if hasattr(func, "underlying_fn"):
                return _TFFn(func.underlying_fn)
            else:
                return _TFFn(None)
        else:
            return _TFFn(tf.function(func))

    @staticmethod
    def is_reloaded_TFFn(obj):
        cls = obj.__class__
        return (
                cls.__name__ == "_UserObject" and
                cls.__module__.startswith("tensorflow.python.saved_model.load")
        )

    def save(self, foldername, filename):
        path = os.path.join(foldername, filename)
        if self.underlying_fn is None:
            tf.saved_model.save(self, path)
        else:
            spec = [tf.TensorSpec([None], tf.float32)
                    for _ in range(self.num_args())]
            concrete = self.underlying_fn.get_concrete_function(*spec)
            tf.saved_model.save(self, path,
                                signatures={'underlying_fn': concrete})

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
            variable_defs: Dict[str, Variable],
            sampling_distribution_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]]
            ],
            interest_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Tensor1[tf32, Samples]
            ],
            estimates_fn: Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]]
            ],
            unknown_param_names: Sequence[str],
            stat_names: Sequence[str],
            known_param_names: Sequence[str],
            transform_on_stats_fn: Optional[Callable[
                [Tuple[Tensor1[tf32, Samples], ...]],
                Dict["str", Tensor1[tf32, Samples]],
            ]],
            transform_on_stats_stat_names: Sequence[str],
            param_sampling_regularize_jitter_multiply: float,
            param_sampling_regularize_jitter_add: float,
            profile: str,
            network_setup_args: Optional[Dict],
            optional_data_to_store: Optional[Dict],
    ) -> None:

        if optional_data_to_store is None:
            optional_data_to_store = {}

        self.sampling_distribution_fn = _TFFn.get(sampling_distribution_fn)
        self.interest_fn = _TFFn.get(interest_fn)
        self.estimates_fn = _TFFn.get(estimates_fn)

        self.unknown_param_names = unknown_param_names
        self.stat_names = stat_names
        self.known_param_names = known_param_names

        self.transform_on_stats_fn = _TFFn.get(transform_on_stats_fn)
        self.transform_on_stats_stat_names = transform_on_stats_stat_names

        self.param_sampling_regularize_jitter_multiply = \
                                    param_sampling_regularize_jitter_multiply
        self.param_sampling_regularize_jitter_add = \
                                    param_sampling_regularize_jitter_add

        self.profile = profile

        self.network_setup_args = network_setup_args
        self.optional_data_to_store = optional_data_to_store
        self.variable_defs = variable_defs

    def kwargs(self) -> Dict:
        return self._wrap_up_kwargs(
            sampling_distribution_fn = self.sampling_distribution_fn,
            interest_fn = self.interest_fn,
            estimates_fn = self.estimates_fn,
            unknown_param_names = self.unknown_param_names,
            stat_names = self.stat_names,
            known_param_names = self.known_param_names,
            transform_on_stats_fn = self.transform_on_stats_fn,
            transform_on_stats_stat_names =
                                self.transform_on_stats_stat_names,
            param_sampling_regularize_jitter_multiply =
                                self.param_sampling_regularize_jitter_multiply,
            param_sampling_regularize_jitter_add =
                                self.param_sampling_regularize_jitter_add,
            profile=self.profile,
            network_setup_args = self.network_setup_args,
            optional_data_to_store = self.optional_data_to_store,
        ) | self.variable_defs

    @staticmethod
    def _wrap_up_kwargs(**kwargs):
        return kwargs

    def save(self, foldername, profile=None) -> None:
        if profile is None:
            profile = self.profile

        # Save tf_fn separately as need to use tf saving stuff for this
        self.sampling_distribution_fn.save(
            foldername, "sampling_distribution_fn",
        )
        self.interest_fn.save(
            foldername, "interest_fn",
        )
        self.estimates_fn.save(
            foldername, "estimates_fn",
        )
        self.transform_on_stats_fn.save(
            foldername, "transform_on_stats_fn",
        )

        # Pickle variable defs separately: need special treatment on loading
        for variable_def in self.variable_defs.values():
            variable_def.deapply_tf_functions()
        self._save_pickle(foldername, "variable_defs", self.variable_defs)
        for variable_def in self.variable_defs.values():
            variable_def.reapply_tf_functions()

        other_args = {"unknown_param_names": self.unknown_param_names,
                      "stat_names": self.stat_names,
                      "known_param_names": self.known_param_names,
                      "transform_on_stats_stat_names":
                                        self.transform_on_stats_stat_names,
                      "profile": profile,
                      "network_setup_args": self.network_setup_args,
                      "optional_data_to_store": self.optional_data_to_store,
                      "param_sampling_regularize_jitter_multiply":
                                self.param_sampling_regularize_jitter_multiply,
                      "param_sampling_regularize_jitter_add":
                                self.param_sampling_regularize_jitter_add}
        self._save_pickle(foldername, "other_args", other_args)

    @classmethod
    def load(
            cls: Type[T],
            foldername: str,
            new_kwargs_for_backward_compatibility: dict,
    ) -> T:
        kwargs = cls._wrap_up_kwargs(
            sampling_distribution_fn = _TFFn.load(
                foldername, "sampling_distribution_fn",
            ),
            interest_fn = _TFFn.load(
                foldername, "interest_fn",
            ),
            estimates_fn = _TFFn.load(
                foldername, "estimates_fn",
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

        if new_kwargs_for_backward_compatibility is not None:
            other_args |= new_kwargs_for_backward_compatibility

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

    def store_data(
            self,
            main_key: str,
            sub_key: str,
            data_dict: Dict,
    ) -> None:

        if main_key not in self.optional_data_to_store:
            self.optional_data_to_store[main_key] = {}
        if sub_key not in self.optional_data_to_store[main_key]:
            self.optional_data_to_store[main_key][sub_key] = {}
        now = str(datetime.now())
        self.optional_data_to_store[main_key][sub_key][now] = data_dict
