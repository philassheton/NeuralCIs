import tensorflow as tf

from neuralcis._simulator_net import _SimulatorNet

from abc import ABC, abstractmethod

from typing import Tuple, Optional
from tensor_annotations import tensorflow as ttf
from tensor_annotations.tensorflow import Tensor1
from neuralcis.common import NetInputBlob, NetTargetBlob
from neuralcis.common import NetInputSimulationBlob, Indices


class _SimulatorNetCached(_SimulatorNet, ABC):
    def __init__(
            self,
            cache_size: int,
            **sim_net_kwargs,
    ) -> None:

        super().__init__(**sim_net_kwargs)
        self.cache_size = cache_size
        self.current_index = tf.Variable(0)
        self.cache = None

    def get_ready_for_training(
            self,
    ) -> None:

        super().get_ready_for_training()
        if self.cache is None:
            self.cache = self.simulate_training_data_cache(self.cache_size)
            random_order = tf.random.shuffle(tf.range(self.cache_size))
            self.cache = self.pick_indices_from_cache(self.cache, random_order)

    @tf.function
    def simulate_training_data(
            self,
    ) -> Tuple[NetInputBlob, Optional[NetTargetBlob]]:

        n = self.batch_size
        indices = (tf.range(n) + self.current_index) % self.cache_size
        self.current_index.assign(indices[-1] + 1)
        simulation_blob, target_blob =  self.pick_indices_from_cache(
            self.cache,
            indices
        )
        net_input_blob, target_blob = self.simulate_data_from_cache_chunk(
            simulation_blob,
            target_blob
        )
        return net_input_blob, target_blob

    @abstractmethod
    def simulate_training_data_cache(
            self,
            n: ttf.int32,
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        pass

    @abstractmethod
    def simulate_validation_data_cache(
            self
    ) -> Tuple[NetInputBlob, NetTargetBlob]:

        pass

    @abstractmethod
    def pick_indices_from_cache(
            self,
            cache: Tuple[NetInputSimulationBlob, NetTargetBlob],
            indices: Tensor1[ttf.int16, Indices],
    ) -> Tuple[NetInputSimulationBlob, NetTargetBlob]:

        pass

    @tf.function
    def simulate_data_from_cache_chunk(
            self,
            input_simulation_blob: NetInputSimulationBlob,
            target_blob: NetTargetBlob,
    ) -> Tuple[NetInputBlob, NetTargetBlob]:

        return input_simulation_blob, target_blob
