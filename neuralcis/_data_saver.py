import tensorflow as tf

from typing import Optional, List, Dict, Sequence
from neuralcis.common import INSTANCE_VARS, WEIGHTS, FILE_PATH


class _DataSaver:
    # subobjects to save maps the filename suffix for the object to the object
    # OR an array of objects
    def __init__(
            self,
            filename: str,
            subobjects_to_save: Optional[Dict] = None,
            instance_tf_variables_to_save: Optional[List] = None,
            net_with_weights_to_save: tf.keras.Sequential = None,
            nets_with_weights_to_save: Sequence[tf.keras.Sequential] = (),
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

        if filename == "":
            return
        else:
            self.load(filename)

    def save(self, filename: str) -> None:
        if filename == "":
            return

        if len(self.nets_with_weights_to_save) == 1:
            weights_filename = self.weights_filename(filename)
            self.nets_with_weights_to_save[0].save_weights(weights_filename)
        elif len(self.nets_with_weights_to_save) > 1:
            for i, net in enumerate(self.nets_with_weights_to_save):
                net.save_weights(self.weights_filename(filename, i))

        for suffix, obj in self.subobjects_to_save.items():
            object_filename = self.construct_filename(filename, suffix)
            print(f'saving {object_filename}')
            obj.save(object_filename)

        if len(self.instance_tf_variables_to_save):
            tf.raw_ops.Save(
                filename=self.instance_variables_filename(filename),
                tensor_names=self.instance_tf_variables_to_save,
                data=[getattr(self, var) for var in
                      self.instance_tf_variables_to_save]
            )

    def load(self, filename: str) -> None:
        if filename == "":
            return

        if len(self.nets_with_weights_to_save) == 1:
            weights_filename = self.weights_filename(filename)
            self.nets_with_weights_to_save[0].load_weights(weights_filename)
        elif len(self.nets_with_weights_to_save) > 1:
            for i, net in enumerate(self.nets_with_weights_to_save):
                net.load_weights(self.weights_filename(filename, i))

        for suffix, obj in self.subobjects_to_save.items():
            object_filename = self.construct_filename(filename, suffix)
            obj.load(object_filename)

        for var in self.instance_tf_variables_to_save:
            value = tf.raw_ops.Restore(
                file_pattern=self.instance_variables_filename(filename),
                tensor_name=var,
                dt=tf.float32
            )
            getattr(self, var).assign(value)

    @staticmethod
    def construct_filename(filename: str, suffix: str) -> str:
        return "%s %s" % (filename, suffix)

    @staticmethod
    def instance_variables_filename(filename: str) -> str:
        filename = _DataSaver.construct_filename(filename, INSTANCE_VARS)
        return _DataSaver.add_path(filename)

    @staticmethod
    def weights_filename(filename: str, index: Optional[int] = None) -> str:
        if index is None:
            weights_name = WEIGHTS
        else:
            weights_name = f'{WEIGHTS} {index}'
        filename = _DataSaver.construct_filename(filename, weights_name)
        return _DataSaver.add_path(filename)

    @staticmethod
    def add_path(filename: str) -> str:
        return "%s/%s" % (FILE_PATH, filename)

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
