import os
from contextlib import contextmanager

from praxis import base_layer

try:
  import transformer_engine.jax as te
  from transformer_engine.common import recipe
  _IS_TRANSFORMER_ENGINE_INSTALLED = True
  DEFAULT_INIT_MUTABLE_LIST = base_layer.DEFAULT_INIT_MUTABLE_LIST + [te.fp8.FP8Helper.FP8_COLLECTION_NAME]

except ModuleNotFoundError as e:
  _IS_TRANSFORMER_ENGINE_INSTALLED = False
  DEFAULT_INIT_MUTABLE_LIST = base_layer.DEFAULT_INIT_MUTABLE_LIST


class TransformerEngineHelperBase:

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="replica", tp_mesh_axis="mdl", fsdp_mesh_axis="data"):
        raise NotImplementedError


class TENotInstalledHelper(TransformerEngineHelperBase):

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="replica", tp_mesh_axis="mdl", fsdp_mesh_axis="data"):
        try:
            yield
        finally:
            pass


class TEInstalledHelper(TransformerEngineHelperBase):

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="replica", tp_mesh_axis="mdl", fsdp_mesh_axis="data"):
        fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID,
                                           amax_history_len=1024, amax_compute_algo='max')

        enable_fp8 = bool(int((os.environ.get("ENABLE_FP8", False))))
        try:
            with te.fp8_autocast(enabled=enable_fp8,
                                 fp8_recipe=fp8_recipe,
                                 mesh_resource=te.MeshResource(dp_resource=dp_mesh_axis,
                                                               tp_resource=tp_mesh_axis,
                                                               fsdp_resource=fsdp_mesh_axis)):
                yield
        finally:
            pass


class TransformerEngineHelper(TransformerEngineHelperBase):

    @staticmethod
    def is_enabled_te():
        enable_te = bool(int((os.environ.get("ENABLE_TE", False))))
        return (_IS_TRANSFORMER_ENGINE_INSTALLED and enable_te)

    @staticmethod
    def get_helper():
        if TransformerEngineHelper.is_enabled_te():
            return TEInstalledHelper
        return TENotInstalledHelper

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="replica", tp_mesh_axis="mdl", fsdp_mesh_axis="data"):
        try:
            with TransformerEngineHelper.get_helper().fp8_autocast(dp_mesh_axis, tp_mesh_axis, fsdp_mesh_axis):
                yield
        finally:
            pass
