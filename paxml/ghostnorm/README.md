# Ghost Norm Clipping

Ghost Norm Clipping refers to a technique to do per-example gradient norm clipping without explicitly materializing the per-example gradient tensors. Per-example gradient norm clipping is required to implement *Differentially Private SGD* (DP-SGD) training. Materializing per-example gradient tensors could lead to prohibitive memory consumption for large models, restricting the maximum batch size that one accelerator chip could handle when training with DP-SGD. Ghost norm clipping avoids this issue by directly computing the per-example gradient norms using the existing information available from a standard back-propagation algorithm.

This algorithm was originally proposed in [Goodfellow (2015)](https://arxiv.org/abs/1510.01799) for fully connected layers, and later on extended to support [convolution](https://arxiv.org/abs/2205.10683) and [attention](https://arxiv.org/abs/2110.05679) layers.

## Ghost Norm Clipping in PAX

Ghost norm clipping is implemented via `jax.custom_vjp` rules. It consists of two parts:

- `paxml.sgf.GhostClippingDpSgdStochasticGradient` is a subclass of `paxml.sgf.BaseStochasticGradient` that contains the logic to use a 2-pass algorithm to implement DP-SGD with ghost norm clipping. A user can use this class in place of `paxml.sgf.DpSgdStochasticGradient` to get ghost norm clipping based DP-SGD.
- `paxml.ghostnorm` contains implementation of custom layers that uses `jax.custom_vjp` to compute per-example gradient norms during back-propagation. To use ghost norm clipping, all the parametric layers (layers with trainable parameters) in the model need to implement this protocol. See `paxml.ghostnorm.base` for more details about the protocol.
