# Copyright 2025 the LlamaFactory team.
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

from transformers import KernelConfig
from typing import TYPE_CHECKING

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def configure_hf_kernels(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
) -> None:
    """
    LLaMA-Factory supports `kernels` to accelerate model training and inference.
    Since the `kernels` framework already natively supports multiple hardware types, no additional hardware detection is required.
    Note that `kernels` is still in active development, so LLaMA-Factory currently only supports a subset of models and kernel types.
    LLaMA-Factory will expand this support as the `kernels` ecosystem matures.
    """

    if not model_args.enable_hf_kernels:
        return

    try:
        model_list = ["llama", "qwen2", "qwen2_vl", "qwen2_5_vl", "qwen3", "qwen3_moe"]
        model_type = getattr(config, "model_type", None)
        if model_type in model_list:
            _KERNELS_MAPPING = {
                "SiLU": {
                    "cuda": "kernels-community/activations:SiLU",
                },
                "RMSNorm": {
                    "cuda": "kernels-community/liger_kernels:LigerRMSNorm",
                    "npu": "kernels-ext-npu/rmsnorm:rmsnorm",
                },
            }
            _kernels_config = KernelConfig(_KERNELS_MAPPING)
            config.kernels_config = _kernels_config
    except Exception as e:
        logger.warning(f"Failed to apply huggingface/kernels: {e}")
        return

    logger.info("huggingface/kernels has been applied to the model.")
