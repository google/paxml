# Ghost Norm Clipping

Ghost Norm Clipping refers to a technique to do per-example gradient norm clipping without explicitly materializing the per-example gradient tensors. Per-example gradient norm clipping is required to implement *Differentially Private SGD* (DP-SGD) training. Materializing per-example gradient tensors could lead to prohibitive memory consumption for large models, restricting the maximum batch size that one accelerator chip could handle when training with DP-SGD. Ghost norm clipping avoids this issue by directly computing the per-example gradient norms using the existing information available from a standard back-propagation algorithm.

This algorithm was originally proposed in [Goodfellow (2015)](https://arxiv.org/abs/1510.01799) for fully connected layers, and later on extended to support [convolution](https://arxiv.org/abs/2205.10683) and [attention](https://arxiv.org/abs/2110.05679) layers.

## Ghost Norm Clipping in PAX

Ghost norm clipping is implemented via `jax.custom_vjp` rules. It consists of two parts:

- `paxml.sgf.GhostClippingDpSgdStochasticGradient` is a subclass of `paxml.sgf.BaseStochasticGradient` that contains the logic to use a 2-pass algorithm to implement DP-SGD with ghost norm clipping. A user can use this class in place of `paxml.sgf.DpSgdStochasticGradient` to get ghost norm clipping based DP-SGD.
- `paxml.ghostnorm` contains implementation of custom layers that uses `jax.custom_vjp` to compute per-example gradient norms during back-propagation. To use ghost norm clipping, all the parametric layers (layers with trainable parameters) in the model need to implement this protocol. See `paxml.ghostnorm.base` for more details about the protocol. Note that ghost clipping does not allow sharing trainable parameters between layers.

### Generic Layer

Currently, we support some basic layers such as Linear and Embedding that implement `jax.custom_vjp` to compute per-example gradient norms efficiently. Further, we also support a generic wrapper layer `paxml.ghostnorm.generic_wrapper.WrappedGhostNorm` that can be used to convert any Pax layer to be compatible with ghost clipping. Note that this layer instantiates per-example gradients. However, due to the structure of the computation, the  per-example gradients only need to be materialized per layer and thus, this can still be much more memory efficient than instantiating per-example gradients for the full model.

To use `paxml.ghostnorm.generic_wrapper.WrappedGhostNorm` in Pax, modify your config as follows:

Original:

```
p = pax_fiddle.Config(SomeLayer, ...)
some_layer = p.Instantiate()
```

Updated:

```
from paxml.ghostnorm.generic_wrapper import GhostNormPaxConfig, WrappedGhostNorm

layer_p = pax_fiddle.Config(SomeLayer, ...)
p = GhostNormPaxConfig(WrappedGhostNorm, layer_tpl=layer_p)
some_layer = p.Instantiate()
```

### Generate Wrapped Models

Ghost norm also provides a utility function to dynamically wrap predefined model configs. See `paxml.ghostnorm.generic_wrapper.generate_wrapped_template`. To utilize this functionality, ensure that every parametric layer in our model is either included in

```
_REPLACE_MAP = {
    praxis_embedding.Embedding: embedding.EmbeddingGhostNorm,
    praxis_linears.Linear: linears.LinearGhostNorm,
    praxis_linears.Bias: linears.BiasGhostNorm,
}
```
or

```
_WRAPPABLE_LAYERS = {
    praxis_normalizations.LayerNorm,
    praxis_normalizations.RmsNorm,
    praxis_attentions.PerDimScale,
    praxis_attentions.CausalDepthwiseConv1D,
    praxis_attentions.AttentionProjection,
    praxis_attentions.CombinedQKVProjectionLayer,
    praxis_transformers.TransformerFeedForwardMoe,
}
```
in `generic_wrapper.py`. `_REPLACE_MAP` is a map of pax layers to ghost norm layers that have the same functionality. Any layer with a custom implementation which is meant to replace a Pax layer should be added to this map and while generating the wrapped model template, the Pax layers will be replaced by their ghostnorm counterparts. For other parametric layers in the model for which the wrapper `WrappedGhostNorm` should be used can be added to the list `_WRAPPABLE_LAYERS`. All layers in this list that appear in the model config will be wrapped using `WrappedGhostNorm`.

Tip: If any parametric layer does not implement `jax.custom_vjp` or is not wrapped using `WrappedGhostNorm`, it will raise a `TypeError: <some_op> requires ndarray or scalar arguments, got <class 'paxml.ghostnorm.base.ParamWithAux'>` in most scenarios. This can be a good way to find all the parametric layers in your model in some instances and these layers can be added to `_WRAPPABLE_LAYERS` until the model is able to train using ghost clipping.

NOTE: Currently, this does not seem to work with layers that use `flax.linen.scan` such as `praxis.layers.Repeat` and `praxis.layers.StackedTransformerRepeated`.

#### An example

Given an already defined experiment, we can add ghost clipping to it as follows. For example, let's look at `BertSpmdL4H128` which is an experiment at `paxml.tasks.lm.params.bert`. In this case, we could generate an equivalent experiment with differential privacy using ghost clipping as

```python
@experiment_registry.register
class DpGhostClippingBertSpmdL4H128(BertSpmdL4H128):
  """Bert-Large, training with DP-SGD."""

  # Privacy parameters are chosen arbitrarily here and should be chosen based
  # on the task and privacy requirements.
  L2_NORM_CLIP = 1.0
  NOISE_MULTIPLIER = 0.5

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    lp = task_p.train.learner
    lp.stochastic_gradient = pax_fiddle.Config(sgf.GhostClippingDpSgdStochasticGradient)
    lp.stochastic_gradient.l2_norm_clip = self.L2_NORM_CLIP
    lp.stochastic_gradient.noise_multiplier = self.NOISE_MULTIPLIER

    # Use a separate embedding layer as weight sharing is not compatible with ghost clipping.
    task_p.model.lm_tpl.separate_embedding_tpl = pax_fiddle.Config(layers.Embedding)
    task_p.model = generic_wrapper.generate_wrapped_template(task_p.model)

    return task_p
```


