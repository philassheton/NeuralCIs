# Simultaneous Confidence Intervals through Neural Networks

Generate simultaneous confidence intervals for almost any set of estimands using neural networks.

This is a work in progress, and does not yet work.  Please return here in a few months' time, when I hope the project will be completed.

## Motivation

The key driver for this project is my automated statistician, which will be hosted at my website, [statsadvice.com](https://statsadvice.com).  The automated statistician guides you through your own statistical output from your own experimental data, and explains what the numbers mean, along the way introducing more intuitive versions of the statistics ("common language effect sizes").  

The problem I encountered was that there were various numbers I wanted the automated statistician to generate on the fly (certain "confidence intervals") that would need excessive amounts of computing power (killing my server, and leaving you waiting for your guided tour) and indeed, several for which I could not find any good method.  I realised that a neural network could be trained to calculate these values, and that once trained, the neural network could do so with relatively little computing power and in a fraction of a second.

## The Rough Idea

This is not simply a case of training a net on a set of inputs and outputs but, rather, the neural network works out the confidence intervals from first principles, based only on a simple simulation that is very easy to write.  Note that the following is currently a fairly terse, vague and technical description.  A better readme will follow as the project develops.

For example, suppose I want to generate Tukey confidence intervals for pairwise comparisons of a set of group means.  This is actually already rather easy without neural networks, but it is a great example that is widely understood.  I can provide the neural network with a simulation that randomly samples group means and standard deviations according to their sampling distribution, and a set of functions that describe how the pairwise differences are calculated from the group means.  The neural network does the rest, and returns a neural network that (within certain bounds on its inputs) can return, for any given set of sample group means and sds, simultaneous Tukey confidence intervals on the pairwise differences.

## Why Use Neural Networks?

The key intuition here is that a neural network is an extremely flexible *and differentiable* multivariate function approximator.  As such, this is not so much an application of neural networks as "artifical intelligence" as it use of a neural network as an extremely flexible regression.

The difficulty with calculating confidence intervals is that they effectively are calculated by searching all the possible "worlds" from which your sample could have been drawn, and finding *all* those which quite reasonably *could* have generated your data.  Statistics is full of all sorts of clever tricks for doing this, for example leveraging symmetries in how the world created your dataset to work backwards from your experimental results to the set of results that might have generated it.  

But for cases that fall outside of those clever tricks, there are only a few general methods, which require a fair bit of number crunching, and are all built on approximations to some degree.  In the case where we want to generate confidence intervals for a set of numbers at the same time, the options are even thinner.

On the other hand, a neural network can be trained to form a compact representation of how a given "real world" will lead to different sorts of possible datasets, all in one, *easily differentiable* package.  Another neural network can learn by searching through the different possible worlds that are encapsulated in this network.  

The current approach to neural CIs (in this project) works by training a whole series of networks in this way, to capture progressively more useful representations of how the world generates data, and finally working backwards by training a network that learns to invert this model to find what worlds could have generated any possible dataset.  By solving this for all possible datasets at once, a great deal of effort can be saved in the long run, since otherwise each confidence interval that is generated must repeat the same laborious work of searching through all the possible worlds that could have created it.

I will add a more detailed, and more intuitive explanation of this as the project develops.