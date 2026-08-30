# Simultaneous Confidence Intervals through Neural Networks

Generate simultaneous confidence intervals for almost any set of estimands using neural networks.

This is a work in progress;  the current version only generates $p$-values on a single parameter of interest.

## The Current Version

This project does not yet have a version number, as the first version is still being finalised.  Version 1.0.0 is almost complete; the code works now, but just needs some finishing touches, and the interface might change in small ways until then.  

Currently you can generate a $p$-value (in the full Version 1.0.0, you will also be able to generate a confidence interval) for a single parameter of interest, in the presence also of nuisance parameters and other "known values", such as sample size.  In the simple example below, we reproduce the one sample $t$-test; estimating the mean of a normal distribution of unknown variance.

In future versions, we hope it will be possible to handle simultaneous confidence intervals on multiple quantities, again in the presence of nuisance variables.  There is still considerable work to be done improving the fit of the networks, and in testing them, so p-values at present do not come with any guarantees of accuracy.

### How to use NeuralCIs

For a very basic example of how to fit a NeuralCIs object, see `examples/t-test/simple/fit_t_test_simple.py`, which contains the following chunks:

1. ```python
   import tensorflow as tf
   import tensorflow_probability as tfp

   import neuralcis as nci
   from neuralcis import Location, Scale, PositiveCount
   ```
   
   You're going to need `tensorflow` (and possibly also `tensorflow-probability`) to write a sampling function, as well as some objects from the `neuralcis` package.

2.  ```python
    def sampling_distribution_fn(mu, sigma, n):
        df = n - 1.
        z = tf.random.normal(tf.shape(mu))
        chi_sq = tfp.distributions.Chi2(df).sample(1)[0, :]
      
        mu_hat = z * sigma / tf.math.sqrt(n) + mu
        sigma_hat = sigma * tf.math.sqrt(chi_sq / df)
      
        return {"mu_hat": mu_hat, "sigma_hat": sigma_hat}
    ```
    Next you need to write a sampling function.  This can accept any number of arguments, and these are the parameters to your distribution.

   Parameter values are passed in for each parameter as a 1D `tf.Tensor` of sampled parameter values, all of the same length.  For each element in these parameter tensors, the sampling function should draw **one** sample from the distribution defined by the corresponding parameters.  The result should then be a set of `tf.Tensor` objects of the same size as each parameter, and wrapped up in a python dictionary.  Note that these statistics have different names than the parameters.  

   In the current version of `neuralcis` you will also need to return exactly as many statistics as there are unknown parameters.  Here `n` will be treated as known *a priori*, so there are two *unknown* parameters input to the function and two statistics returned.
   
   *The sampling function must be Tensorflow compatible*.  In the above example, `tf.random.normal` is used to sample from a normal distribution and `tfp.distributions.Chi2(df).sample` to sample from a chi-squared distribution.  The package `tensorflow-probability` contains a huge array of Tensorflow-compatible statistical sampling options.

3.  ```python
    def interest_fn(mu, sigma, n):
        return mu
    ```

    The `interest_fn` computes the parameter of interest from the same parameter inputs as taken by the `sampling_distribution_fn`.

4.  ```python
    def estimates_fn(mu_hat, sigma_hat, n):
        return {"mu": mu_hat, "sigma": sigma_hat}
    ```
   
    The `estimates_fn` maps the statistics returned by `sampling_distribution_fn` plus the known parameters back to estimates of the parameters input to it.  These are **not** used in the main pivoting network, but help the random sampling of parameters to find its way around space and to estimate the determinant of the Fisher information, again used as part of the parameter sampling process.

   Most importantly, these are the basis on which the set of valid inputs to the network are defined.  In step 5 you will specify a bounding "estimates box" around these estimates; the net will attempt to generate valid $p$-values for any sample whose estimates land inside this box.

5.  ```python
    cis = nci.NeuralCIs(
        sampling_distribution_fn,
        interest_fn,
        estimates_fn,
   
        mu=nci.Param(Location(), (-3., 3.)),
        sigma=nci.Param(Scale(), (0.333, 3.)),
   
        n=nci.KnownParam(PositiveCount(), (3., 100.)),
   
        interest=nci.Interest(Location()),
   
        mu_hat=nci.Stat(Location()),
        sigma_hat=nci.Stat(Scale()),
   
        param_sampling_regularize_jitter_add=0.1,
    )
    ```
    
    Finally you are ready to construct your `NeuralCIs` object.  The arguments to this are the `sampling_distribution_fn`, `interest_fn` and `estimates_fn` plus a series of definitions for each of the input and output variables for those functions.  Each of these is one of `Param`, `KnownParam`, `Stat` or `Interest`, depending on the variable's role in the model.  It may also be a `StatCanonical` (see next section).  
   
   The first argument to each of these variable definitions is a `VariableType`, which defines what sort of value it is.  This can currently be any of `Location`, `Scale`, `PositiveCount`, `Correlation` or `Proportion` and defines how the variable will be transformed before it is fed into the net; for example, a `Scale` variable will be log transformed.  These objects have their own properties (for example, it is also possible to set `lowish_highish` values on each that rescale the transformed values to help keep variables on different scales manageable for the network).  So it can be important to construct a separate `VariableType` object for each `Variable`.

   The `Param` and `KnownParam` variable definitions take a second argument: a pair of float values, which define the range of values for this variable.  For a `KnownParam`, this is simply a minimum and maximum value; the known parameter will then be randomly sampled between these two values.  For a `Param`, this is a little more subtle: it is a range of *estimates* values for which we would like $p$-values to be valid.  At inference-time, we will not know whether our unknown parameters are within a given range, but we will be able to run the statistics through the `estimates_fn`.  The training aims to make $p$-values (and in particular, for phase 2, confidence intervals) valid for any sample whose estimates land between the minimum and maximum values defined here.  We refer to the combined valid estimates intervals across the different unknown parameters as the "estimates box".

   Finally, the last argument to `NeuralCIs` is `param_sampling_regularize_jitter_add=0.1`.  This compensates to some extent for the fact that the Jeffreys Prior currently used can overemphasise small $\sigma$ values in this model, in cases like this, where the estimates box is not a single point.  We hope to improve the parameter sampling in the near future.


6.  ```python
    cis.fit()
    ```

    Calling `fit` on this object will run the training.  This takes a few hours on a mid-level laptop GPU.

7.  ```python
    cis.save('saved_model')
    ```
    
    It is a good idea at this point to save the model, so you don't have to rerun the training! The net weights, along with the various functions and variable definitions used to construct the model are here saved in a folder called `'saved_model'`.  You can load it back up any time with `NeuralCIs.load('saved_model')`. 

8.  ```python
    cis.p_and_ci(mu_hat=3.182446/2., sigma_hat=1., null=0., n=4)
    Out[5]: {'p': 0.05001378059387207}
    ```

    It is then possible to generate p-values (confidence intervals are temporarily disabled) for any set of parameters within the ranges defined in step 3.  Into `cis.p_and_ci`, we pass in all statistics and known parameters, plus a null value for our interest parameter.  This case should give a $p$-value of exactly 0.05 and it is not far wrong!

While this example shows just a more complicated way to do something that can already be done on the back of a cigarette packet, the beauty of the neural method is that it can be applied to any quantity that you can write a sampling function for.

### Adding group invariances

The model fitted above treats each `mu` and `sigma` as a separate problem that may have its own properties.  But of course we know that this problem is invariant under scale and transformation.  We can incorporate this by adding canonicalization rules to our parameters and statistics as follows.

First, it will be helpful to see how the code looks if we remove *just the scale invariance*; we will then remove the translation invariance in a second example.

```python
cis = nci.NeuralCIs(
    sampling_distribution_fn,
    interest_fn,
    estimates_fn,

    mu=nci.Param(Location(), (-3., 3.), "mu / sigma_hat"),
    sigma=nci.Param(Scale(), (1., 1.), "sigma / sigma_hat"),

    n=nci.KnownParam(PositiveCount(), (3., 100.), "n"),

    interest=nci.Interest(Location(), "interest / sigma_hat"),

    mu_hat=nci.Stat(Location()),
    sigma_hat=nci.Stat(Scale()),
   
    d_hat=nci.StatCanonical(Location(), "mu_hat / sigma_hat"),
)
```

The main change here, compared with the previous section, is that every `Param`, `KnownParam` and `Interest` has a small piece of Python code wrapped in a string as a third argument.  This snippet of code may refer only to the statistics and to the variable itself (e.g. on the `mu` variable it may refer to `mu`).  Its purpose is to remove invariances, based on the statistics available (doing so based on unknown parameters would present us with problems at inference time).  In this example, each code snippet divides by `sigma_hat` to remove the scale invariance.

We use a slightly different interface for the `Stat` variables, because removing the group action based on the statistics has the effect of canonicalizing the statistics and, in doing so, reduces their dimension.  So we require further variables to be defined to capture each of the remaining dimensions *after canonicalization*.  Here, removing scale by dividing by `sigma_hat` reduces the dimension from two (`mu_hat`, `sigma_hat`) to one.  This one remaining dimension is effectively a Cohen's $d$, so we call it `d_hat`, and it also has a piece of code as a constructor argument, to define how it is computed from the original statistics.

The code above, then removes just scale invariance.  If we were to remove both scale and translation, there would be no remaining dimensions in our statistics, so need no `StatCanonical` variables at all:

```python
cis = nci.NeuralCIs(
    sampling_distribution_fn,
    interest_fn,
    estimates_fn,

    mu=nci.Param(Location(), (0., 0.), "(mu - mu_hat) / sigma_hat"),
    sigma=nci.Param(Scale(), (1., 1.), "sigma / sigma_hat"),

    n=nci.KnownParam(PositiveCount(), (3., 100.), "n"),

    mu_hat=nci.Stat(Location()),
    sigma_hat=nci.Stat(Scale()),

    interest=nci.Interest(Location(), "(interest - mu_hat) / sigma_hat"),
)
```

Note also that the estimates box (`(0., 0.)` and `(1., 1.)`) has been reduced to a single point, because, after canonicalization, *every* estimated `mu` will be zero and every estimated `sigma` will be one.  Because our estimates box has been reduced to a point, we also don't need to add regularizing jitter.

This is the model used in `examples/t-test/canonical`.

## The rough idea

The architecture behind NeuralCIs is based on the concept of a "normalizing flow", which is trained to represent a probability distribution by mapping samples from that distribution to normally distributed outputs.  Our architecture breaks this flow into two components, one which is blind to those inputs we do not know, and another that tries to "fill in the blanks" of the first net to make it (as close as possible) a complete representation of the target probability distribution.

The beauty here is that one need only have a simulation of the underlying probability distribution and NeuralCIs will do the rest.

## Why use neural networks?

The key intuition here is that a neural network is an extremely flexible *and differentiable* multivariate function approximator.  As such, this is not so much an application of neural networks as "artificial intelligence" as it use of a neural network as an extremely flexible regression.  (Though one might argue that all AI really is an incredibly fancy regression, of course.)

The difficulty with calculating confidence intervals is that they effectively are computed by searching all the possible "worlds" from which your sample could have been drawn, and finding *all* those which quite reasonably *could* have generated your data.  Statistics is full of all sorts of clever tricks for doing this, for example leveraging symmetries in how the world created your dataset to work backwards from your experimental results to the set of results that might have generated it.  

But for cases that fall outside those clever tricks, there are only a few general methods, which require a fair bit of number crunching, and are all built on approximations to some degree..

On the other hand, a neural network can be trained to form a compact representation of how *all* possible "real world"s will lead to different sorts of possible datasets, all in one, *easily differentiable* package.  Because it is so readily differentiated, another neural network can learn by searching through the different possible worlds that are encapsulated in this network.  By solving this for all possible datasets at once, a great deal of effort can be saved in the long run, since otherwise each confidence interval that is generated must repeat the same laborious work of searching through all the possible worlds that could have created it.

I will add a more detailed, and more intuitive, explanation of this as the project develops.