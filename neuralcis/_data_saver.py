import os
import tensorflow as tf

from neuralcis._sequential_net import _SequentialNet

from typing import Optional, List, Sequence, Dict
from neuralcis.common import INSTANCE_VARS, SEQUENTIAL


class _DataSaver:
    # subobjects to save maps the filename suffix for the object to the object
    # OR an array of objects
    def __init__(
            self,
            subobjects_to_save: Optional[Dict] = None,
            instance_tf_variables_to_save: Optional[Sequence[str]] = None,
            net_with_weights_to_save: Optional[_SequentialNet] = None,
            nets_with_weights_to_save: Sequence[_SequentialNet] = (),
    ) -> None:

        if subobjects_to_save is None:
            subobjects_to_save = {}
        if instance_tf_variables_to_save is None:
            instance_tf_variables_to_save = []

        nets_with_weights_to_save = list(nets_with_weights_to_save)

        if net_with_weights_to_save is not None:
            nets_with_weights_to_save.append(net_with_weights_to_save)

        self.nets_with_weights_to_save = nets_with_weights_to_save
        self.instance_tf_variables_to_save = instance_tf_variables_to_save
        self.subobjects_to_save = \
            self.preprocess_arrays_into_multiple_dict_elems(subobjects_to_save)

    @staticmethod
    def fullname(
            foldername: str,
            filename: str,
    ) -> str:

        return os.path.join(foldername, filename)

    def save(
            self,
            foldername: str,
            filename_start_internal: str,
    ) -> None:

        fullname_start = self.fullname(foldername, filename_start_internal)
        os.makedirs(foldername, exist_ok=True)

        for i, net in enumerate(self.nets_with_weights_to_save):
            net.save(self.sequential_filename(fullname_start, i))

        for suffix, obj in self.subobjects_to_save.items():
            object_filename = self.construct_filename(filename_start_internal,
                                                      suffix)
            print(f'saving {object_filename}')
            obj.save(foldername, object_filename)

        if len(self.instance_tf_variables_to_save):
            tf.raw_ops.Save(
                filename=self.instance_variables_filename(fullname_start),
                tensor_names=self.instance_tf_variables_to_save,
                data=[getattr(self, var) for var in
                      self.instance_tf_variables_to_save]
            )

    def load(
            self,
            foldername: str,
            filename_start_internal: str,
    ) -> None:

        fullname_start = self.fullname(foldername, filename_start_internal)

        for i, net in enumerate(self.nets_with_weights_to_save):
            net.load(self.sequential_filename(fullname_start, i))

        for suffix, obj in self.subobjects_to_save.items():
            object_filename = self.construct_filename(filename_start_internal,
                                                      suffix)
            obj.load(foldername, object_filename)

        for var in self.instance_tf_variables_to_save:
            value = tf.raw_ops.Restore(
                file_pattern=self.instance_variables_filename(fullname_start),
                tensor_name=var,
                dt=tf.float32
            )
            getattr(self, var).assign(value)

    @staticmethod
    def construct_filename(filename: str, suffix: str) -> str:
        return "%s %s" % (filename, suffix)

    @staticmethod
    def instance_variables_filename(filename: str) -> str:
        return _DataSaver.construct_filename(filename, INSTANCE_VARS)

    @staticmethod
    def sequential_filename(filename: str, index: int) -> str:
        sequential_name = f'{SEQUENTIAL} {index}'
        return _DataSaver.construct_filename(filename, sequential_name)

    @staticmethod
    def preprocess_arrays_into_multiple_dict_elems(dictionary: Dict) -> Dict:
        expanded_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, list):
                for i in range(len(value)):
                    expanded_dict["%s%d" % (key, i)] = value[i]
            else:
                expanded_dict[key] = value
        return expanded_dict
