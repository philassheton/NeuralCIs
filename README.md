# Simultaneous Confidence Intervals through Neural Networks

Generate simultaneous confidence intervals for almost any set of estimands using neural networks.

This is a work in progress;  the current version only handles a simple (and somewhat trivial) case.  Please return here in a few months' time, when I hope the project will be completed.  

## The Current Version

This project does not yet have a version number, as the first version is still being finalised.  Version 1.0.0 is almost complete; the code works now, but just needs some finishing touches, and the interface might change in small ways until then.  

In Version 1.0.0, you will be able to generate p-values and confidence intervals for any single quantity whose sampling distribution depends only on one unknown parameter.  Known parameters, such as sample size, can also be included.  In the simple example below, confidence intervals are calculated for the mean of a normally distributed variable with known variance, given the observed (estimated) mean value and the sample size.  

In future versions (and **this is the real motivator of this project**), it will be possible to handle simultaneous confidence intervals on multiple quantities, also in the presence of nuisance variables (initial experiments in this direction are in the `experimental` branch of the project, but may not be stable or even functioning).

There is still considerable work to be done improving the fit of the networks, and in testing them, so p-values and CIs at present do not come with any guarantees of accuracy.

### How to use NeuralCIs

For a very basic example of how to fit a NeuralCIs object, see simple_mean_of_normal_example.py, which contains the following chunks:

1. ```python
   import tensorflow as tf
   from neuralcis import NeuralCIs, Uniform, LogUniform
   ```
   
   You're going to need `tensorflow` (and/or `tensorflow-probability`) to write a sampling function, as well as some objects from the `neuralcis` package.

2. ```python
   def normal_sampling_fn(mu, sigma, n):
       std_normal = tf.random.normal(tf.shape(mu))
       mu_hat = std_normal * sigma / tf.math.sqrt(n) + mu
       return {"mu": mu_hat}
   ```
   Next you need to write a sampling function.  This can take in any number of arguments, and these are the parameters to your distribution.  In the present version, all but one must be "known parameters", with the one unknown parameter being the one whose value is to be estimated.

   Parameter values are passed in for each parameter as a 1D `Tensor` of sampled parameter values.  The function should sample from the sampling distribution for the estimate of the unknown parameter, once for each parameter sample passed in.  

   It returns a dict mapping the name of the unknown parameter (*must be the same as the name used in the function signature*) to the estimate of its value.  In later versions, it will be possible to have multiple parameters estimated, but at present this dict should include only one entry.
   
   *The sampling function must be Tensorflow compatible*.  In the above example, `tf.random.normal` is used to sample from a normal distribution.  The package `tensorflow-probability` also contains a huge array of Tensorflow-able statistical sampling options.

3. ```python
   cis = NeuralCIs(
       normal_sampling_fn,
       mu=Uniform(-2., 2.),
       sigma=LogUniform(.1, 10.),
       n=LogUniform(3., 300.)
   )
   ```
   
   Now you can construct a NeuralCIs object, passing to the constructor your sampling distribution function, as well as a series of named arguments that define how each parameter is sampled.  These must have the same names as the inputs to the sampling function, and their values must be subclasses of the `neuralcis.Distribution` class.  Here `mu` is sampled uniformly between -2 and 2; `sigma` is sampled from a log-uniform distribution, with the lowest possible value for sigma being 0.1, and the highest being 10.  At present, these are the two sampling distributions that are available.


4. ```python
   cis.fit()
   ```
   
   Calling `fit` on this object will run the training.

5. ```python
   print(cis.p_and_ci(1.96, mu=0., sigma=4., n=16.))
   ```
   
   It is then possible to generate p-values and confidence intervals for any set of parameters within the ranges defined in step 3.  The first value here is the observed (estimated) value of `mu`, while the value of `mu` is the null value, under which the *p*-value is calculated.

   Here, I have asked for a *p*-value and CI for a sample mean of 1.96 in a sample of size 16, with known standard deviation of 4, and null hypothesis that `mu = 0`.  This is a very straightforward case, where we should expect a *p*-value of 0.05, and a CI of [0, 3.92] (where the upper bound is just 1.96 + 1.96).  The neural network doesn't get it perfect, but for practical purposes this would certainly be close enough:

   ```python
   {'p': 0.0516, 'mu_lower': -0.0312, 'mu_upper': 3.8253}
   ```


While this example shows just a more complicated way to do something that can already be done on the back of a cigarette packet, the beauty of the neural method is that it can be applied to any quantity that you can write a sampling function for.  Read on for more about the bigger picture...

## Motivation

The key driver for this project is my automated statistician, which will be hosted at my website, [statsadvice.com](https://statsadvice.com).  The automated statistician guides you through your own statistical output from your own experimental data, and explains what the numbers mean, along the way introducing more intuitive versions of the statistics ("common language effect sizes", and their confidence intervals).  

The problem I encountered was that there were various numbers I wanted the automated statistician to generate on the fly (particularly certain confidence intervals) that would need excessive amounts of computing power (killing my server, and leaving you waiting for your guided tour) and indeed, several for which I could not find any good method.  I realised that a neural network could be trained to calculate these values, and that once trained, the neural network could do so with relatively little computing power and in a fraction of a second.

## The Rough Idea

This is not simply a case of training a net on a set of inputs and outputs but, rather, the neural network works out the confidence intervals from first principles, based only on a simple simulation that is very easy to write.  Note that the following is currently a fairly terse, vague and technical description.  A better readme will follow as the project develops.

For example, suppose I want to generate Tukey confidence intervals for pairwise comparisons of a set of group means.  This is actually already rather easy without neural networks, but it is a great example that is widely understood.  I can provide the neural network with a simulation that randomly samples group means and standard deviations according to their sampling distribution, and a set of functions that describe how the pairwise differences are calculated from the group means.  The neural network does the rest, and returns a neural network that (within certain bounds on its inputs) can return, for any given set of sample group means and sds, simultaneous Tukey confidence intervals on the pairwise differences.

## Why Use Neural Networks?

The key intuition here is that a neural network is an extremely flexible *and differentiable* multivariate function approximator.  As such, this is not so much an application of neural networks as "artificial intelligence" as it use of a neural network as an extremely flexible regression.

The difficulty with calculating confidence intervals is that they effectively are calculated by searching all the possible "worlds" from which your sample could have been drawn, and finding *all* those which quite reasonably *could* have generated your data.  Statistics is full of all sorts of clever tricks for doing this, for example leveraging symmetries in how the world created your dataset to work backwards from your experimental results to the set of results that might have generated it.  

But for cases that fall outside those clever tricks, there are only a few general methods, which require a fair bit of number crunching, and are all built on approximations to some degree.  In the case where we want to generate confidence intervals for a set of numbers at the same time, the options are even thinner.

On the other hand, a neural network can be trained to form a compact representation of how all possible "real world"s will lead to different sorts of possible datasets, all in one, *easily differentiable* package.  Because it is so readily differentiated, another neural network can learn by searching through the different possible worlds that are encapsulated in this network.  

The current approach to neural CIs (in this project) works by training a whole series of networks in this way, to capture progressively more useful representations of how the world generates data, and finally working backwards by training a network that learns to invert this model to find what worlds could have generated any possible dataset.  By solving this for all possible datasets at once, a great deal of effort can be saved in the long run, since otherwise each confidence interval that is generated must repeat the same laborious work of searching through all the possible worlds that could have created it.

I will add a more detailed, and more intuitive, explanation of this as the project develops.