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

from abc import ABC
import fiddle as fdl

from paxml import tasks_lib

from praxis import pax_fiddle
from praxis.layers import transformers

from praxis.contrib.gpu.scripts_gpu.lora_layers import (
    LoraAttentionProjection,
    LoraCombinedQKVProjection,
    LoraLinear,
)

class LoRAMixin(ABC):
    USE_LORA = False
    LORA_RANK = 8
    LORA_TARGET_LAYERS = "all"

    def _validate(self):
        if self.LORA_TARGET_LAYERS not in ["all", "attention", "mlp"]:
            raise ValueError(
                "LAYERS_TO_INCLUDE_FOR_LORA should be one of all, attention or mlp."
            )

    def configure_lora(
        self, task_p: pax_fiddle.Config[tasks_lib.SingleTask]
    ) -> pax_fiddle.Config[tasks_lib.SingleTask]:
        if not self.USE_LORA:
            return task_p

        self._validate()
        train_p = task_p.train

        if hasattr(self, "CHECKPOINT_IGNORE_RULES"):
            self.CHECKPOINT_IGNORE_RULES = [r"^.*lora.*$"]

        train_p.learner.bprop_variable_inclusion = [r"^.*lora.*$"]
        stacked_p = task_p.model.lm_tpl.stacked_transformer_tpl
        if issubclass(
            fdl.get_callable(stacked_p), transformers.StackedTransformerRepeated
        ):
            stacked_p = stacked_p.block
        stacked_p = stacked_p.transformer_layer_params_tpl

        if self.LORA_TARGET_LAYERS in ["all", "mlp"]:
            ff_templ = stacked_p.tr_fflayer_tpl.fflayer_tpl
            original_linear_p = ff_templ.linear_tpl
            ff_templ.linear_tpl = pax_fiddle.Config(
                LoraLinear,
                rank=self.LORA_RANK,
                name="lora_linear",
            )
            ff_templ.linear_tpl.copy_fields_from(original_linear_p)

        if self.LORA_TARGET_LAYERS in ["all", "attention"]:
            if hasattr(stacked_p.tr_atten_tpl, "combined_qkv_proj_tpl"):
                original_combined_qkv_p = stacked_p.tr_atten_tpl.combined_qkv_proj_tpl
                stacked_p.tr_atten_tpl.combined_qkv_proj_tpl = pax_fiddle.Config(
                    LoraCombinedQKVProjection,
                    name="lora_qkv_projection",
                    rank=self.LORA_RANK,
                )
                stacked_p.tr_atten_tpl.combined_qkv_proj_tpl.copy_fields_from(
                    original_combined_qkv_p
                )

            original_proj_p = stacked_p.tr_atten_tpl.proj_tpl
            stacked_p.tr_atten_tpl.proj_tpl = pax_fiddle.Config(
                LoraAttentionProjection,
                name="lora_attention_projection",
                rank=self.LORA_RANK,
            )
            stacked_p.tr_atten_tpl.proj_tpl.copy_fields_from(original_proj_p)

        return task_p
