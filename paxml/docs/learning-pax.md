# Learning Pax

doc/pax/learning-pax

[TOC]

## About Learning Pax

This page provides a path for learning Pax. It lists the prerequisites as well
as a brief introduction to JAX, Flax, and some key Pax components. It also
includes pointers to resources where you can learn more.

Adjacent to this page is [Key Concepts][concepts]; a description of additional
concepts that you will surely encounter when working with Pax.

Once you have consumed these articles, you should have a pretty good idea of
what you need to learn.

You should also consider working through the [Hands-on Tutorals][tutorials]; a
set of SME-created Jupyter Notebooks designed to get you comfortable working with Pax.


## Prerequisites

To work with Pax, you are expected to be familiar with the following concepts
and technologies:

### Machine Learning

**Machine Learning (ML)** is software system that builds (aka *trains*) a
predictive model from input data. The system uses this learned model to make
predictions from new (never-before-seen) data.

### Neural Network

**Neural Networks**—a subset of machine learning—is a programming construct
composed of layers of nodes (aka *neurons*). Each node contains a numerical
value, which is modified during training. In **Deep Learning**, layers of nodes
feed their resulting values into subsequent layers of nodes. A full-trained
model of these nodes can be then used to make predictions on new data.

**Recommended Resources:**

*   [First Neural Network for Beginners Explained][nn-article]

### Transformers

**Transformers** are the state-of-the-art architecture for a wide variety of
language model applications, such as text prediction, language translation,
and text generation.

The full Transformer model consists of an **encoder** and a **decoder**. The
encoder converts input text into an intermediate representation, and the decoder
uses that intermediate representation to predict some text.

Although Transformers were initially used for language models, they can also be
applied to different modalities, including speech, image, and video.

Transformer models can be extremely large. Many of the models that require Pax's multipod
capabilities are Transformer models.

**Recommended resources:**

*   [Attention is All You Need][attention] - The 2017 Paper that introduces the
    concept of *self-attention*, one of the key concepts of Transformer models.

### Linear Algebra

**Linear algebra** is the branch of mathematics that deals with the theory of
systems of linear equations, matrices, determinants, vector spaces, and linear
transformations: all of which are used extensively in working with neural
networks.

Two common linear algebra operations you will frequently encounter include:

*   *Matrix multiplication* - used when applying weights to the nodes in a
    layer.
*   *Dot product* - used when you want a measure of how close two vectors are.

TIP: Practice linear algebra until you are comfortable using it! Linear algebra
is not particularly difficult to learn and is extremely valuable in any machine
learning work you do.

**Recommended resources:**

*   [Linear Algebra for Beginners][la-vid] - YouTube Video

### Calculus

**Calculus** is the branch of mathematics that deals with finding derivatives
and integrals of functions.

In machine learning, differential calculus is used for finding the minimal loss
during training. It often uses the *Gradient Descent* algorithm of determining a
local minimum value.

Since there are now libraries—such as *Autograd*—that can easily be used to
automatically differentiate Python and NumPy code, the need for calculus in the
early stages of working with neural network models is not as important as it
once was.

As you advance in this area, you are likely to find other practical uses for
calculus. But that is out of scope of this overview.

**Recommended resources:**

*   [Calculus I][cal-vid] - YouTube Video series

## Learning JAX

To effectively work with Pax, you need to understand and be comfortable using
JAX. The following is an overview of JAX highlights as well as pointers to
learning resources.

JAX is a machine learning framework that transforms numerical functions in order
to speed up machine learning tasks.

*How does it do this?*

At its core, JAX is a language for expressing and composing transformations of
numerical programs. The speeding up of the tasks is done via these composable
functions; transforming the models and data into a form suitable for parallelism
across GPU and TPU accelerator chips.

If you are already familiar with Python's NumPy library, then you are off to a
good start for learning JAX.

For Pax, the key JAX functions are:

Operation | Purpose
--------- | -------------------------------------------------
`grad`    | Automatic differentiation
`jit`     | Re-compilation into a JAX-compatible function
`pjit`    | Re-compilation to work effectively with GPUs/TPUs
`pmap`    | SPMD programming
`vmap`    | Auto-vectorization

The following provides a brief introduction to each of these concepts:

### `grad`

JAX uses an updated version of **Autograd**: a library of tools that can
automatically differentiate native Python and NumPy functions. It can
differentiate through loops, branches, recursion, and closures, and it can even
take derivatives of derivatives of derivatives. `grad` is capable of both
reverse-mode differentiation (also known as *backpropagation*) and forward-mode
differentiation; and the two can be composed arbitrarily to any order.

### SPMD

**Single program, multiple data (SPMD) programming** refers to a parallelism
technique where the same computation is run on different input data in parallel
on different devices. The goal of SPMD is to obtain results faster. It is the
most common style of parallel programming.

### `vmap`

`vmap` (short for *vectorizing map*) returns a function which maps its input
function over argument axes. It is similar to `map`, although instead of keeping
the loop on the outside, it pushes the loop down into a function’s primitive
operations for better performance.

**Auto-vectorization** is a concept in parallel computing where a computer
program is converted:

*   *from* a **scalar implementation** - which processes a single pair of
    operands at a time
*   *to* a **vector implementation** - which processes one operation on multiple
    pairs of operands at once.

### `pmap`

`pmap` (short for *parallel map*) takes a user function, and execute copies of
the function over multiple underlying hardware devices (CPUs, GPUs, or TPUs),
*with different input values*, via SPMD.

### `pjit`

`pjit` enables users to shard computations without rewriting them by using the
SPMD partitioner. The user passes a function to `pjit`, which returns a function
that has the equivalent semantics but is compiled into an *XLA computation* that
runs across multiple devices (such as GPUs or TPU cores).

### The `pmap` and `pjit` Paradigms

With the `pmap` paradigm, the model weights and optimizer states are fully
replicated across all TPU cores. As a result, this mode only support data
parallelism.

With the `pjit` paradigm, the model weights and optimizer states are sharded
across different TPU cores. So this mode provides support for model parallelism.
However, it demands more effort on the user side by requiring sharding
annotations and the specification of a mesh shape and mesh axes.

**Recommended resources:**

*   JAX Read the Docs: [Tutorial: JAX 101][jax-101]
*   [Lab 1: A JAX Primer][jax-primer] - Jupyter Notebook
*   [Lab 2: JAX for edit-distance][jax-ed] - Jupyter Notebook
*   [Parallel Evaluation in JAX][parallel-eval]

TIP: JAX is amazing and will prove to be extremely useful throughout your Pax
journey. You will not regret learning it!

## Learning Flax

Flax is a high-performance neural network library for JAX that is designed for
flexibility. In other words, Flax has extended JAX to add neural network
capabilities. Pax borrows heavily from the Flax library.

**Recommended resources:**

*   [Flax Fundamentals][flax-vid] - Recording of presentation. **Highly
    recommended!**
*   Pax Workshop: [Jupyter Notebook #3: Pax Layer Basics][pax-layers] - Jupyter Notebook

## Pax Elements

Pax is made up of several basic components that are common to most neural
network frameworks, including *Layers*, *Models*, *Tasks*, and *Experiments*.

Let's take a brief look at each of these:

### Layer

The layer is the fundamental building block in deep learning. It is a set of
nodes, each of which holds a numerical value.

A neural network model is composed of numerous layers that feed into each other.
A layer can be thought of as a *container* of nodes, each of which has a
numerical value, receives weighted inputs, transforms it with a set of (mostly)
non-linear functions, and then passes these values as output to the next layer.

A 'Pax layer' represents an arbitrary function, possibly with trainable
parameters. They inherit from the Flax nn.Module. A layer can contain other
layers as children.

A Pax layer always inherits from base_layer.BaseLayer (which internally inherits
from Flax nn.Module). All non-trivial Pax layers have 3 parts: `HParams`,
`setup`, and `__call__`.

Pax uses a **Layer** object that, by default, is implemented using the
**Praxis** layer library. While Praxis is optimized for ML at scale, Praxis has
a goal to be usable by other JAX-based ML projects.

To see what you can do with Pax Layers, check out [Layers How-to's][layers]
page.

### Model

A model is a collection of layers defining the network.

In Pax, the **Model** object defines the network; typically a collection of
Layers and the interfaces for interacting with the model, such as decoding.

Some example [base models][model-code] include:

*   LanguageModel
*   SequenceModel
*   ClassificationModel

To see what you can do with Pax Models, check out [Models How-to's][models]
page.

### Task

A **Task** is an object that defines everything you need to accomplish your
project's goal. ML tasks include goals such as: regression, classification,
clustering, machine translation, anomaly detection, etc.

A Task is typically a combination of:

*   Model
*   Optimizer
*   Metrics aggregator (optional)

To see what you can do with Pax Tasks, check out [Tasks How-to's][tasks] page.

### HParam

**Hyperparameters** are a set of values that control the neural networks's
learning process.

Pax uses **HParam** to declare all of its models' hyperparameters in one place.

`HParam` is a dataclass responsible for configuration. They can be
systematically inspected and compared. This makes them suitable for debugging,
and troubleshooting.

Note: The configuration portion of HParams, is in the process of being replaced
by Fiddle.

To see what you can do with Pax HParams, check out [HParams How-to's][hparams]
page.

### Experiment

An **Experiment** object defines all of the ingredients you need (such as
hyperparameter values, weights, data, etc.) to make your project *reproducible*.
You can *register* your experiments so that you can visualize and understand
the models (and their runs) on tools such as [TensorBoard][tensorboard].

An Experiment is the combination of a concrete 'Task' and an associated DataSet
that is used to train and evaluate the model.  It typically involves running the
same pre-defined computation in different hyperparameter configurations. Each
experiment is usually associated with a resource allocation.

A Pax development scenario might include:

*   Model - Create the model
*   Task - Create the Params and Task
*   Register - Register the experiment
*   Cloud - Run on Cloud
*   TensorBoard - Watch it converge

To see what you can do with Pax Experiments, check out [Life of a Pax
Experiment][life] and [Experiments How-to's][experiments].

## The Pax Code

The Pax code itself is considered a canonical example of how to assemble Pax
Shared Components into a training codebase. See:

*   [Praxis][praxis-code]: The Core high-performance ML library of Pax (often
    referred to as the "Layer library").
*   [PaxML][paxml-code]: The library to combine all the ML blocks together.

### Praxis

Note: Praxis is a separate library from PaxML because, although Praxis been
created as Pax's Layer library—specifically optimized for machine learning at
scale—it has been designed to be usable by *other* JAX-based ML projects as
well.

**Praxis** contains not just the definitions for the Layer class, but most of
its supporting components as well. This includes data inputs, configuration
libraries (HParam and Fiddle), optimizers, etc. Praxis provides the definitions
for the Model class, which is built on top of the Layer class.

Additionally, Praxis also contains dozens of pre-defined layers. These can be used directly or as "model
code" for new layers.

### PaxML

The **PaxML** can be thought of as the infrastructure and "glue" code for
necessary for putting all of the Pax components. Among other things it includes
the definitions for tasks, experiments, evaluating (including decoding),
checkpoints, metrics, AutoML, SeqIO and more.

## Example Code

For an example, this is the Pax implementation of the [ResNet
layers][resnet-layers], where you can see how setup() and fprop() are
defined. The model is implemented and registered as follows: [ResNet
Model][resnet-model]. Here you can specify the dataset and task for the model.

To get started use the [Pax Basics Tutorial][pax-layers]
where you can see some example implementations of Pax layers.

For a more comprehensive intro in Pax, take a look at the Pax 101
[colabs][pax-101-colab]


<!-- Reference Links -->

[attention]: https://arxiv.org/abs/1706.03762
[cal-vid]: https://www.youtube.com/playlist?list=PLmvTkFr4eZzQRLBGzbc5sZokmP-PqyNSJ
[concepts]: https://github.com/google/paxml/tree/main/paxml/docs/concepts.md
[experiments]: https://github.com/google/paxml/tree/main/paxml/docs/experiments.md
[hparams]: https://github.com/google/paxml/tree/main/paxml/docs/hparams.md
[jax-101]: https://jax.readthedocs.io/en/latest/jax-101/index.html
[jax-ed]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax101_jax_for_edit_distance.ipynb
[jax-primer]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax101_jax_primer.ipynb
[la-vid]: https://www.youtube.com/watch?v=kZwSqZuBMGg
[layers]: https://github.com/google/paxml/tree/main/paxml/docs/layers.md
[life]: https://github.com/google/paxml/tree/main/paxml/docs/life_of_an_experiment.md
[model-code]: https://github.com/google/praxis/tree/main/praxis/layers/models.py
[models]: https://github.com/google/paxml/tree/main/paxml/docs/models.md
[nn-article]: https://towardsdatascience.com/first-neural-network-for-beginners-explained-with-code-4cfd37e06eaf
[parallel-eval]: https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html
[pax-101-colab]: https://github.com/google/paxml/tree/main/paxml/docs/pax_101_ipynbs.md
[pax-layers]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax_layer_basics.ipynb
[paxml-code]: https://github.com/google/paxml/tree/main/paxml
[praxis-code]: https://github.com/google/praxis/tree/main/praxis
[resnet-layers]: https://github.com/google/praxis/tree/main/praxis/layers/resnets.py
[resnet-model]: https://github.com/google/paxml/tree/main/paxml/tasks/vision/params/imagenet_resnets.py
[tasks]: https://github.com/google/paxml/tree/main/paxml/docs/tasks.md
[tensorboard]: https://www.tensorflow.org/tensorboard
[tutorials]: https://github.com/google/paxml/tree/main/paxml/docs/hands-on-tutorials.md

