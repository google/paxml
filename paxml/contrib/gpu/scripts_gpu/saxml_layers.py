# coding=utf-8
# Copyright 2022 The Pax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Customize layers for sax.

Copied from
https://github.com/google/saxml/blob/main/saxml/server/pax/lm/layers.py
"""

from typing import Optional, Tuple

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import embedding_softmax

template_field = base_layer.template_field
JTensor = pytypes.JTensor
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
NestedMap = py_utils.NestedMap


class LLaMARotaryEmbedding(embedding_softmax.RotaryPositionalEmbedding):
  """LLaMA variant of ROPE where inputs are split in a different way."""

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: JTensor,
      position: Optional[JTensor] = None,
  ) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a JTensor of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    if len(inputs.shape) != 4:
      raise ValueError(
          'Input is assumed to be a rank 4 tensor of shape'
          '[batch, sequence, heads, dims].'
      )
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          'The embedding dims of the rotary position embedding'
          'must match the hidden dimension of the inputs.'
      )
    inputs_shifted_left = jnp.concatenate(
        [inputs[..., 1:], inputs[..., :1]], axis=-1
    )
    inputs_shifted_right = jnp.concatenate(
        [inputs[..., -1:], inputs[..., :-1]], axis=-1
    )
    inputs_shifted = jax.lax.select(
        jnp.tile(
            jnp.mod(jnp.arange(self.embedding_dims, dtype=jnp.int32), 2),
            inputs.shape[:-1] + (1,),
        ),  # [[[[0, 1, 0, 1, ...], ...]
        inputs_shifted_right,
        inputs_shifted_left,
    )
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    fraction = jnp.repeat(fraction, 2)
    timescale = (
        self.min_timescale
        * (self.max_timescale / self.min_timescale) ** fraction
    )
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    sign = jnp.sign(
        jnp.mod(jnp.arange(self.embedding_dims, dtype=jnp.int32), 2) - 0.5
    )  # [-1, 1, -1, 1, ...]
    outputs = inputs * cos + inputs_shifted * sin * sign
    if self.cast_as_fprop_dtype:
      outputs = outputs.astype(self.fprop_dtype)
    return outputs


class FakeLayerNorm(layers.LayerNorm):

  def setup(self) -> None:
    return

  def __call__(self, inputs, paddings=None):
    return inputs


class TransformerMLP(layers.TransformerFeedForward):
  ln_tpl: LayerTpl = template_field(FakeLayerNorm)


# TODO(huangyp): adapt the more efficient lingvo implementation.
class ParallelTransformer(layers.Transformer):
  """Transformer with parallel attention and feedforward."""

  norm_policy = 'pre'  # Use primer_hybrid for GPT-Neo
  residual_droppath_prob = 0.0
  use_cross_attention = False
  tr_fflayer_tpl: LayerTpl = template_field(TransformerMLP)

  def ffn_norm(self, inputs: JTensor, inputs_normalized: JTensor) -> JTensor:
    # Apply FFN layer
    if self.norm_policy == 'primer_hybrid':
      ffn_inputs = self.post_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      ffn_inputs = inputs_normalized
    else:
      ffn_inputs = inputs
    return ffn_inputs

  def __call__(
      self,
      inputs: JTensor,
      paddings: JTensor,
      attention_mask: JTensor,
      cross_inputs: Optional[JTensor] = None,
      cross_attention_mask: Optional[JTensor] = None,
      segment_pos: Optional[JTensor] = None,
      segment_ids: Optional[JTensor] = None,
  ) -> Tuple[JTensor, JTensor]:
    """Transformer decoder layer for GPT-J and NeoX.

    Args:
      inputs: Input sequence JTensor of shape [B, T, H].
      paddings: Input paddings JTensor of shape [B, T] (only used in FFN layer).
      attention_mask: Self attention mask ready to add to the logits. It can be
        of shape [1|B, 1, 1|T, T] which is broadcast compatible with the self
        attention matrix of shape [B, N, T, T]. This is assumed to have combined
        paddings, causal masking as well as segment maskings.
      cross_inputs: Output of the encoder, to be used for cross attention, of
        shape [B, S, H].
      cross_attention_mask: Cross attention mask ready to add to the logits. It
        can be of shape [1|B, 1, 1|T, S] which is broadcast compatible with the
        cross attention matrix of shape [B, N, T, S]. This is assumed to have
        combined paddings as well as segment maskings.
      segment_pos: A JTensor of shape [B, T]. The position of each token in a
        segment.
      segment_ids: A JTensor of shape [B, T] specifying which segment each token
        belongs to.

    Returns:
      The fflayer output with shape [B, T, D].
      atten_probs: A NestedMap with keys `self_atten` <float>[B, N, T, T].
    """
    assert not self.use_cross_attention
    assert self.residual_droppath_prob == 0.0
    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)
    else:
      inputs_normalized = inputs
    # Compute self-attention, key/value vectors are the input itself
    atten_output, self_atten_probs = self.self_attention(
        inputs_normalized,
        inputs_normalized,
        inputs_normalized,
        atten_mask=attention_mask,
        query_segment_pos=segment_pos,
        key_segment_pos=segment_pos,
    )
    atten_probs = NestedMap(self_atten=self_atten_probs)

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)

    ffn_inputs = self.ffn_norm(inputs, inputs_normalized)
    ffn_output = self.ff_layer(ffn_inputs, paddings=paddings)
    output = atten_output + ffn_output + inputs
    return output, atten_probs  # pytype: disable=bad-return-type  # jax-ndarray

  def extend_step(
      self,
      inputs: JTensor,
      *,
      time_step: JTensor,
      segment_pos: Optional[JTensor] = None,
      attention_mask: JTensor,
      cross_attention_mask: Optional[JTensor] = None
  ) -> JTensor:
    # pyformat:disabled
    """Transformer decoder layer, autoregressive cached decoding.

    For cross attention, the key/value cache may have a smaller batch size b
    than inputs batch size B. In this case, we require B % b == 0, and this
    corresponds to multi-sample decoding for each input in b, and cross-
    attention states will be repeated by (B // b) times. Each consecutive
    (B // b) chunk in B correspond to multiple samples for the same cross
    # inputs.

    When `inputs` has shape [B, D], it will do extend_step on one token per
    batch in regular autoregressive decoding.

    When `inputs` has shape [B, L, D], it will do extend_step on L tokens per
    batch. This is used to do suffix scoring after autoregressive decoding.

    Args:
      inputs:         [B, D] or [B, L, D], target sequence at index time_step.
      time_step:      a 0-based scalar, the current decode step.
      segment_pos:    [B] or [B, L], the current position in the same segment.
        If unspecified, time_step will be used.
      attention_mask: [B, 1, L, S] if extends multiple steps (i.e. `inputs` is
        of shape [B, L, D]) or [B, 1, T] if extends one step (i.e. `inputs` is
        of shape [B, D]), optional attention mask for this time step. This
        combines causal mask with any segment mask if applicable.
      cross_attention_mask: [b|B, 1, 1 S], optional, cross_segment_mask for this
        time step. This combines padding mask with any segment mask if
        applicable.

    Returns:
      output: [B, D] or [B, L, D].
    """
    # Layer normalize input
    if self.norm_policy == 'primer_hybrid':
      inputs_normalized = self.pre_layer_norm(inputs)
    elif self.norm_policy == 'pre':
      inputs_normalized = self.layer_norm(inputs)
    else:
      inputs_normalized = inputs

    # Self-attention layer.
    atten_output = self.self_attention.extend_step(
        inputs_normalized,
        atten_mask=attention_mask,
        time_step=time_step,
        segment_pos=segment_pos,
    )

    # Residual dropout and connection
    atten_output = self.residual_dropout(atten_output)
    # Apply FFN layer
    ffn_inputs = self.ffn_norm(inputs, inputs_normalized)
    ffn_output = self.ff_layer.extend_step(ffn_inputs, time_step=time_step)
    output = atten_output + ffn_output + inputs
    return output


class ParallelTransformerOnlyNormAttentionInputs(ParallelTransformer):
  """Transformer with parallel attention and feedforward."""

  def ffn_norm(self, inputs: JTensor, inputs_normalized: JTensor) -> JTensor:
    # Apply FFN layer
    if self.norm_policy == 'primer_hybrid':
      ffn_inputs = self.post_layer_norm(inputs)
    else:
      ffn_inputs = inputs_normalized
    return ffn_inputs


class GPTJRotaryEmbedding(embedding_softmax.RotaryPositionalEmbedding):
  """GPTJ variant of ROPE where rotary_dim != dim_per_head."""

  max_position_embeddings: Optional[int] = None
  rotary_dim: Optional[int] = None

  def setup(self) -> None:
    super().setup()
    self.embed_positions = self.create_sinusoidal_positions(
        self.max_position_embeddings, self.rotary_dim
    )

  def create_sinusoidal_positions(self, num_pos, dim):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    sinusoid_inp = jnp.einsum(
        'i , j -> i j', jnp.arange(num_pos), inv_freq
    ).astype('float32')
    sin, cos = jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)
    return jnp.concatenate((sin, cos), axis=-1)

  def rotate_every_two(self, tensor):
    rotate_half_tensor = jnp.stack(
        (-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), axis=-1
    )
    rotate_half_tensor = rotate_half_tensor.reshape(
        rotate_half_tensor.shape[:-2] + (-1,)
    )
    return rotate_half_tensor

  def apply_rotary_pos_emb(self, tensor, sin_pos, cos_pos):
    sin_pos = sin_pos[:, :, None, :].repeat(2, 3)
    cos_pos = cos_pos[:, :, None, :].repeat(2, 3)
    return (tensor * cos_pos) + (self.rotate_every_two(tensor) * sin_pos)

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: JTensor,
      position: Optional[JTensor] = None,
  ) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a JTensor of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    if len(inputs.shape) != 4:
      raise ValueError(
          'Input is assumed to be a rank 4 tensor of shape'
          '[batch, sequence, heads, dims].'
      )

    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.int32)[jnp.newaxis, :]

    sincos = jnp.take(self.embed_positions, position, axis=0)
    sincos = jnp.split(sincos, 2, axis=-1)
    sin, cos = sincos
    inp_rot = inputs[:, :, :, : self.rotary_dim]
    inp_pass = inputs[:, :, :, self.rotary_dim :]
    inp_rot = self.apply_rotary_pos_emb(inp_rot, sin, cos)
    first_part = inp_rot
    second_part = inp_pass
    if self.cast_as_fprop_dtype:
      first_part = first_part.astype(self.fprop_dtype)
      second_part = second_part.astype(self.fprop_dtype)
    return jnp.concatenate([first_part, second_part], axis=-1)
