# Layers

doc/pax/layers

[TOC]

## Introduction

Layers are the building blocks for Pax models.

At a fundemental level, a layer represents an arbitrary function—typically with
trainable parameters. In Pax, Layers can contain other Layers as children,
allowing for some fairly complex structures.

A Layer typically represents a subset of your network (which is defined in its
entirety by the Pax model). You can build your own layers from scratch. That
said, as with any mature framework, the Pax codebase implements a [plethora of
basic layers][plethora] that you can combine into interesting ways to build the
next breakthrough in AI.

The Pax framework supports (and encourages) component re-use. So, share away!

> Tip: For a rudamentary introduction to the basic Pax components, check out
> [Pax Elements][pax-elements]. If you want to dive in for a hands-on
> experience, try the [Pax Layer Basics Jupyter Notebook][layer_ipynb].

## Layer How-To's

### Define a Layer

In Pax, Layers inherit from `BaseLayer` which, in turn, inherits from the Flax
`nn.Module`. The definition of the Base Layer can be found in
[base_layer.py][base-layer].

If you look at the definition of the `BaseLayer` class, you can see that
subclasses are expected to override the following functions:

*   `setup()`
*   `__call__()`

```python
class Linear(base_layer.BaseLayer):
  ...

  def setup(self):
    self.create_variable('w', WeightHParams(shape=[p.input_dims, ...

  def __call__(self, inputs):
    return jnp.matmul(inputs, self.theta.w)
```

#### Configuration

Every layer will need to have its parameters configured to initial values.

When you look at the class definition for `BaseLayer`, you will see that
`HParams()` should also be overridden in order to define the layer's
configuration. HParams for configuration is in the process of being replaced by
the Fiddle configuration library; cf: Pax [Upcoming feature-25][upcoming-feature-25] and
[Upcoming feature-25a][upcoming-feature-25a].

Tip: We recommend that you configure your layer using
[Fiddle](internal-link/fiddle).


#### `setup`

The purpose of this method is to *set up* your layer. This includes creating the
layer's variables (such as the trainable weights), as well as instantiating all
of its child sub-layers.

`setup` declares the layer variables/weights using `self.create_variable`.  The
methods `self.create_child` and `self.create_children` are used when making
complex networks involving sub-layers.

For example, the following declares a trainable weight `w`. Note that Pax
requires users to statically annotate the shape and dtype of the weight, in
addition to the usual initializer method.

```
self.create_variable(
    'w',
    WeightHParams(
        shape=[p.input_dims, p.output_dims],
        init=p.params_init,
        dtype=p.dtype))
```

Trainable weights can be accessed as `self.theta.w`.

Note: In an ML context, the word **theta** is often used in conjunction with the
*trainable* variables.

The following declares a *non-trainable* weight `moving_mean`.
`REQUIRES_MEAN_SYNC` tells the training loop to sync the mean of this variable
after train step, which you can ignore for now.

```
mva = WeightHParams(
    shape=[p.output_dims],
    init=WeightInit.Constant(0.0),
    dtype=p.dtype,
    collections=[base_layer.WeightHParamsCollection.REQUIRES_MEAN_SYNC])
self.create_variable(
  'moving_mean',
  mva,
  trainable=False)
```

Non-trainable weights can be accessed via `self.get_var('moving_mean')`.

Here's how to declare a sublayer `linear`:

```
def setup(self):
  p = self.hparams
  linear_layer_p = p.linear_tpl.clone()
  linear_layer_p.set(
      input_dims=p.input_dims,
      output_dims=p.output_dims)
  self.create_child('linear', linear_layer_p)
```

Sublayer can be referred to as `self.linear`.

#### `__call__`

This method is the main method that carries out ML computation.

As data moves through each layer in the network, it will be applied to that
layer's `__call__()` method, where the mathematics behind the model is
implemented. This is the *forward-pass* (or *forward propagation*) computation.

*   `self.theta.w` refers to trainable weight `w`
*   `self.get_var('moving_mean')` refers to non-trainable weight `moving_mean`
*   Trainable weights are immutable in `__call__` while non-trainable weights
    can be updated with `self.update_var('moving_mean', new_moving_mean)`
*   Sublayer `__call__` can be expressed as: \
    `projected_inputs = self.linear(inputs)`

(FYI: The code for *back pass* (or *back propagation*) will come from the
specific optimizer being used for training.)

### Select an Existing Layer

For a list of existing layers: https://github.com/google/praxis/tree/main/praxis/layers/

### Report Summaries

Users may want to report summaries to be shown in TensorBoard. Inside layer
`__call__`, users can do so with `self.add_summary`. Example:

```
self.add_summary(
  'inputs_mean', jnp.mean(inputs))
```

### Introspect How Variable Collections are Tracked

WARNING: Do not use `ffn.__call__` directly, you will get a mysterious error
about try_setup.



<!-- Reference Links -->

[base-layer]: https://github.com/google/praxis/tree/main/praxis/base_layer.py
[ipynb_pax_layer]: https://github.com/google/paxml/tree/main/paxml/docs/tutorials/pax_layer_basics.ipynb
[layer_ipynb]: https://github.com/google/paxml/tree/main/paxml/docs/hands-on-tutorials.md#pax-layer-basics
[models]: https://github.com/google/paxml/tree/main/paxml/docs/models.md
[pax-elements]: https://github.com/google/paxml/tree/main/paxml/docs/learning-pax.md#pax-elements
[plethora]: https://github.com/google/praxis/tree/main/praxis/layers
[upcoming-feature-25]: internal-link/pax-upcoming-feature-25
[upcoming-feature-25a]: internal-link/pax-upcoming-feature-25a
