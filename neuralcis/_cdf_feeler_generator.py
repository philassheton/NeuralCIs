import tensorflow as tf
import numpy as np
from datetime import datetime
from neuralcis._sampling_feeler_generator import _SamplingFeelerGenerator
from neuralcis import common

# typing
from typing import Callable, Tuple
from neuralcis.common import Samples, Estimates, Params, UnknownParams
from neuralcis.common import MinAndMax, ImportanceIngredients
from tensor_annotations.tensorflow import Tensor1, Tensor2, Tensor3
from tensor_annotations import tensorflow as ttf

tf32 = ttf.float32
NUM_IMPORTANCE_INGREDIENTS = 2
NetInputSimulationBlob = Tuple[
    Tensor2[tf32, Samples, Params],                           # Centroid
    Tensor3[tf32, Samples, UnknownParams, UnknownParams],     # Cholesky factor
]
NetTargetBlob = Tensor2[tf32, Samples, ImportanceIngredients]


###############################################################################
#
#  TODO: Currently experimenting with Anderson-Darling statistic to see if it
#        can give us better averaging across samples than the ideal
#        Kolmogorov-Smirnov statistic
#
###############################################################################

class _CDFFeelerGenerator(tf.keras.Model):
    def __init__(
            self,
            sampling_distribution_fn: Callable[
                [Tensor2[tf32, Samples, Params]],  # params
                Tensor2[tf32, Samples, Estimates],  # -> ys
            ],
            sampling_feeler_generator: _SamplingFeelerGenerator,
            p_fn: Callable[
                [Tensor2[tf32, Samples, Estimates],
                 Tensor2[tf32, Samples, Params]],
                Tensor1[tf32, Samples]
            ],
            num_samples_per_param: int = \
                                      common.CDF_FEELER_SAMPLES_PER_TEST_PARAM,
            num_params_per_batch: int = common.CDF_FEELER_PARAMS_PER_BATCH,
            subsample_proportion: float = 1.,
    ) -> None:

        super().__init__()
        self.sampling_distribution_fn = sampling_distribution_fn
        self.sampling_feeler_generator = sampling_feeler_generator
        self.p_fn = p_fn
        self.num_samples_per_param = num_samples_per_param
        self.num_params_per_batch = num_params_per_batch
        self.subsample_proportion = subsample_proportion

        # valid = sampling_feeler_generator.valid_row_indices()
        # num_param_samples, = valid.shape
        #
        # if subsample_proportion < 1.:
        #     print("WARNING: subsample_proportion option in "
        #           " _CDFFeelerGenerator is currently for debugging only!!"
        #           "  It does not correctly adjust the chols for the reduced"
        #           " sample size.")
        #     num_param_samples = int(num_param_samples * subsample_proportion)
        #
        # lost_samples = num_param_samples % num_params_per_batch
        # if lost_samples > 0:
        #     num_param_samples -= lost_samples
        #     if subsample_proportion == 1.:
        #         print(f"WARNING: you will waste {lost_samples}/"
        #               f"{num_param_samples}"
        #               f" ({int(lost_samples/num_param_samples * 100)}%) of"
        #               f" your param samples at this batch size.  It is"
        #               f" inevitable with the current design that a few will be"
        #               f" wasted, but please check this is a fairly small"
        #               f" proportion!!")
        # valid = tf.random.shuffle(valid)[:num_param_samples]
        # params = tf.gather(sampling_feeler_generator.sampled_params,
        #                    valid, axis=0)
        # self.sampling_feeler_indices = valid
        # self.num_param_samples = num_param_samples
        #
        # self.darlings = tf.Variable(tf.fill((num_param_samples, ), np.nan),
        #                             dtype=tf.float32)
        # self.darlings_filled = tf.Variable(0, dtype=tf.int64)
        #
        # dataset = tf.data.Dataset.from_tensor_slices(params)
        # self.dataset = dataset.batch(num_params_per_batch, drop_remainder=True)
        # self.compile(optimizer="sgd", loss=None)
        #
        # # PHIL!! Just for experiments
        # self.ks = tf.Variable(tf.fill((num_param_samples, ), np.nan),
        #                             dtype=tf.float32)



    @tf.function
    def train_step(self, data):
        params = data
        params_repeated = tf.repeat(params, self.num_samples_per_param, axis=0)
        estimates = self.sampling_distribution_fn(params_repeated)
        ps = self.p_fn(estimates, params_repeated)
        ps_per_param = tf.reshape(ps, (self.num_params_per_batch,
                                       self.num_samples_per_param))
        cdfs = tf.sort(ps_per_param, axis=1) + 1e-10
        ni = self.num_samples_per_param
        darlings = -ni - tf.reduce_sum(
            (2. * tf.range(ni, dtype=tf.float32) + 1) / ni *
            (tf.math.log(cdfs) + tf.math.log(tf.reverse(cdfs, axis=(1,)))),
            axis=1,
        )
        fill_to = self.darlings_filled + self.num_params_per_batch
        self.darlings[self.darlings_filled:fill_to].assign(darlings)


        # PHIL!!
        ks = tf.reduce_max(tf.math.abs(
            tf.linspace(0., 1., self.num_samples_per_param)[None, :] - cdfs
        ), axis=1)
        self.ks[self.darlings_filled:fill_to].assign(ks)



        self.darlings_filled.assign(fill_to)
        return {}

    def fit(self, *args, **kwargs):




        # PHIL!!  Taken out of constructor!!  Try to make this a little tidier
        #       or find a better place for it

        valid = self.sampling_feeler_generator.valid_row_indices()
        num_param_samples, = valid.shape

        if self.subsample_proportion < 1.:
            print("WARNING: subsample_proportion option in "
                  " _CDFFeelerGenerator is currently for debugging only!!"
                  "  It does not correctly adjust the chols for the reduced"
                  " sample size.")
            num_param_samples = int(num_param_samples *
                                    self.subsample_proportion)

        lost_samples = num_param_samples % self.num_params_per_batch
        if lost_samples > 0:
            num_param_samples -= lost_samples
            if self.subsample_proportion == 1.:
                print(f"WARNING: you will waste {lost_samples}/"
                      f"{num_param_samples}"
                      f" ({int(lost_samples/num_param_samples * 100)}%) of"
                      f" your param samples at this batch size.  It is"
                      f" inevitable with the current design that a few will be"
                      f" wasted, but please check this is a fairly small"
                      f" proportion!!")
        valid = tf.random.shuffle(valid)[:num_param_samples]
        params = tf.gather(self.sampling_feeler_generator.sampled_params,
                           valid, axis=0)
        self.sampling_feeler_indices = valid
        self.num_param_samples = num_param_samples

        self.darlings = tf.Variable(tf.fill((num_param_samples, ), np.nan),
                                    dtype=tf.float32)
        self.darlings_filled = tf.Variable(0, dtype=tf.int64)

        dataset = tf.data.Dataset.from_tensor_slices(params)
        self.dataset = dataset.batch(self.num_params_per_batch,
                                     drop_remainder=True)
        self.compile(optimizer="sgd", loss=None)

        # PHIL!! Just for experiments
        self.ks = tf.Variable(tf.fill((num_param_samples, ), np.nan),
                                    dtype=tf.float32)







        # PHIL!!  Does this help?
        self.sampling_feeler_generator.release_gpu()



        assert len(args) == 0
        assert len(kwargs) == 0
        print(f"{datetime.now()} -- Computing CDF distances from uniform on"
              f" {self.num_param_samples} param samples split into"
              f" {self.num_params_per_batch}")
        super().fit(self.dataset, epochs=1)