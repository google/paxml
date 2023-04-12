# About Pax

doc/pax/about-pax

[TOC]

## What is Pax?

Pax is a programming framework designed for training large-scale neural network
models; models so large that they span multiple TPU accelerator chip slices or
pods.

> NOTE: **TPU pods** are contiguous set of TPU trays grouped together with a
> specialized network. Pod sizes depend on the TPU version. **TPU slices** are
> sub-sections of a pod, and can be scheduled via Cloud. For more on TPUs, see
> [TPU Basics][tpu-basics].

## Pax Goals

Pax is designed to *maximize velocity of ML research and deployment*.

To this end, Pax has been focusing on *scale efficiently*.

**Scaling**

*   **The Pax Approach:** Pax achieves this via a codebase specifically
    built for scaling – with recipes and reference examples for large model
    training, including across modalities (such as text, vision, speech, etc).


### Backend

##### JAX

Pax is built on *JAX*.

**JAX** is a machine learning framework that transforms numerical functions in
order to speed up machine learning tasks.

NOTE: For a *very* basic JAX introduction, with pointers to learning resources,
see the [Learning JAX][learning-jax] section of [Learning Pax][learning-pax]
page.

JAX is built using *Autograd* and *XLA*.

##### Autograd

**Autograd** is an automatic differentiation library that abstracts the
complicated mathematics and calculates gradients of high-dimensional curves with
only a few lines of code.

##### XLA

**Accelerated Linear Algebra** (XLA) is an execution engine that compiles linear
algebra equations to run efficiently on specified acceleration chips (typically
GPUs or TPUs). As a result, it can deliver supercomputer performance through a
simple API.

### Pax Components

Pax itself heavily relies on the libraries and components of the JAX ML
ecosystem.

##### SeqIO

Pax uses the **SeqIO** library *for processing sequential data* to be fed into
downstream sequence models.

##### Optax

**Optax** is a library of composable tools for gradient-based *optimization* of
deep neural networks. (*Optimizers* are used to efficiently reduce the error of
a model by updating the values of each layer's weights and biases.)

##### Fiddle

**Fiddle** is a *configuration* library well-suited to ML applications.

##### Orbax & TensorStore

**Orbax** is a *checkpointing* library for efficiently reading and writing large
multi-dimensional arrays. These libraries use **TensorStore** for their
formatting. (See orbax-checkpoint and [TensorStore][tensorstore]
documentation.)

##### Flax

**Flax** is a high-performance *neural network* library built on JAX. Pax is
built on the Flax infrastructure.

##### PyGlove

**PyGlove** is a library for AutoML and advanced ML programming, with a seamless
integration with Vizier. It powers all tuning and AutoML scenarios
on PAX.

### Pax Elements

Pax is made up of several objects that are common to most neural network
frameworks. These include:

*   **Layers** - A set of "neurons" (or nodes) that hold numeric values and is
    key to any neural network. In Pax, they are implemented via the **Praxis**
    layer library.
*   **Models** - A collection of layers defining the network.
*   **Hyperparameters** - In a neural network, the parameters that control the
    learning process. In Pax, they are implemented through **HParams** (though
    it will soon be replaced by the **Fiddle** configuration system).
*   **Tasks** - Everything you need to accomplish your project's goal. ML tasks
    include goals such as: regression, classification, clustering, machine
    translation, anomaly detection, etc.
*   **Experiments** - All of the ingredients you need (such as hyperparameter
    values, weights, data, etc.) to make your project reproducible. You
    *register* your experiments so that they can be accessed by XManager and so
    you can visualize and understand their runs and models on tools such as
    TensorBoard.

##### Terminology

**Pax Shared Components** - The Pax library distribution is the set of
components, libraries, and API’s that make up Pax. The Pax library distribution
is built in such a way that allows for exploration and adaptation while still
establishing standards and useful layers.

The Pax elements contribute to the **Pax codebase**. Additionally, this codebase
is a canonical way to put the Pax library distribution together to create useful
tools for training large multipod models.

**On Pax** refers to using the *Pax codebase* + the *Pax Shared Components* that
contribute to Pax.

<!-- Reference Links -->

[learning-jax]: https://github.com/google/paxml/tree/main/paxml/docs/learning-pax.md#learning-jax
[learning-pax]: https://github.com/google/paxml/tree/main/paxml/docs/learning-pax.md
[model-code]: https://github.com/google/praxis/tree/main/praxis/layers/models.py
[tensorstore]: https://google.github.io/tensorstore/
[tpu-basics]: https://cloud.google.com/tpu/docs/training-on-tpu-pods
