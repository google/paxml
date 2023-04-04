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

"""Tests for GPT-3 models defined in c4.py."""
import os

from absl.testing import absltest
import fiddle as fdl
import jax
from jax.lib import xla_bridge
from paxml.tasks.lm.params import c4
from praxis import layers
from praxis import optimizers
from praxis import schedules
from praxis import test_utils

prev_xla_flags = None


def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (
        flags_str + " --xla_force_host_platform_device_count=768"
    )
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()


def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


class C4Test(test_utils.TestCase):

  def test_gpt3_mlperf_bs1p5k_config(self):
    config = c4.C4SpmdPipelineGpt3AdamMLPerfHPBS1p5k768Replicas()
    task_p = config.task()

    # Model architecture
    lm_tpl = task_p.model.lm_tpl
    self.assertEqual(config.MAX_SEQ_LEN, 2048)
    self.assertEqual(config.NUM_LAYERS, 96)
    self.assertEqual(config.NUM_HEADS, 96)
    self.assertEqual(lm_tpl.model_dims, 12288)
    self.assertEqual(config.HIDDEN_DIMS, 12288 * 4)
    self.assertGreaterEqual(lm_tpl.vocab_size, 50257)
    self.assertEqual(
        lm_tpl.position_emb_tpl.cls,
        layers.embedding_softmax.TrainablePositionalEmbedding,
    )

    global_batch_size = int(
        config.PERCORE_BATCH_SIZE * jax.device_count() + 1e-6
    )
    self.assertEqual(global_batch_size, 1536)
    self.assertEqual(
        task_p.train.eval_interval_steps * global_batch_size, 24 * 1024
    )

    # Early stopping fn
    self.assertEqual(task_p.early_stopping_fn.cls, c4.EarlyStoppingFn)
    self.assertAlmostEqual(task_p.early_stopping_fn.target_log_pplx, 2.69)

    # optimizer and HPs
    optimizer_p = task_p.train.learner.optimizer
    self.assertEqual(fdl.get_callable(optimizer_p), optimizers.Adam)
    self.assertAlmostEqual(optimizer_p.weight_decay, 0.1)
    self.assertAlmostEqual(optimizer_p.epsilon, 1e-8)
    self.assertAlmostEqual(optimizer_p.beta1, 0.9)
    self.assertAlmostEqual(optimizer_p.beta2, 0.95)
    self.assertAlmostEqual(optimizer_p.clip_gradient_norm_to_value, 1.0)

    # LR schedule
    lr_schedule = optimizer_p.lr_schedule
    self.assertEqual(lr_schedule.cls, schedules.LinearRampupCosineDecay)
    self.assertEqual(lr_schedule.warmup_steps * global_batch_size, 265 * 1536)
    self.assertEqual(lr_schedule.decay_start, lr_schedule.warmup_steps + 1)
    self.assertEqual(lr_schedule.decay_end * global_batch_size, 108600 * 1536)
    self.assertAlmostEqual(optimizer_p.learning_rate, 2e-5)
    self.assertAlmostEqual(lr_schedule.min_ratio, 0.1)
    self.assertAlmostEqual(lr_schedule.max, 1.0)


if __name__ == "__main__":
  absltest.main()
