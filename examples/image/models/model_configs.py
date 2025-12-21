# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from models.unet import UNetModel
from models.vit import DiT_FM_models, DiTFlowMatching

MODEL_CONFIGS = {
    "imagenet": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 3,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "imagenet_discrete": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.2,
        "channel_mult": [2, 3, 4, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "cifar10": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "cifar10_discrete": {
        "in_channels": 3,
        "model_channels": 96,
        "out_channels": 3,
        "num_res_blocks": 5,
        "attention_resolutions": [2],
        "dropout": 0.4,
        "channel_mult": [3, 4, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}

# DiT model configurations for different datasets
# Note: input_size should match the image size used during training
DIT_MODEL_CONFIGS = {
    "imagenet": {
        "input_size": 64,  # For 64x64 resized images (can use 32 for faster training)
        "in_channels": 3,
        "num_classes": 1000,
        "class_dropout_prob": 0.1,  # Model-level dropout for classifier-free guidance
        "learn_sigma": False,
    },
    "cifar10": {
        "input_size": 32,  # CIFAR-10 is 32x32
        "in_channels": 3,
        "num_classes": 10,  # CIFAR-10 has 10 classes
        "class_dropout_prob": 0.1,  # Model-level dropout for classifier-free guidance
        "learn_sigma": False,
    },
}


def instantiate_model(
    architechture: str, 
    is_discrete: bool, 
    use_ema: bool,
    model_type: str = "unet",
    dit_model: str = "DiT-S/2",
    class_dropout_prob: float = None,
    image_size: int = None,
) -> Union[UNetModel, DiscreteUNetModel, DiTFlowMatching]:
    """
    Instantiate a model for flow matching training.
    
    Args:
        architechture: Dataset name (imagenet, cifar10)
        is_discrete: Whether to use discrete flow matching
        use_ema: Whether to wrap model with EMA
        model_type: Model type - 'unet' or 'vit' (DiT)
        dit_model: DiT model variant (e.g., 'DiT-S/2', 'DiT-B/2', etc.)
        class_dropout_prob: Override class dropout probability for classifier-free guidance
        image_size: Image size for DiT models (must match training data size)
    """
    if model_type == "vit":
        assert not is_discrete, "DiT/ViT models do not support discrete flow matching yet."
        assert dit_model in DiT_FM_models, f"Unknown DiT model: {dit_model}. Available: {list(DiT_FM_models.keys())}"
        
        # Get base config for the dataset
        if architechture in DIT_MODEL_CONFIGS:
            config = DIT_MODEL_CONFIGS[architechture].copy()
        else:
            # Default config
            config = {
                "input_size": 32,
                "in_channels": 3,
                "num_classes": 1000,
                "class_dropout_prob": 0.1,
                "learn_sigma": False,
            }
        
        # Override class dropout if specified
        if class_dropout_prob is not None:
            config["class_dropout_prob"] = class_dropout_prob
        
        # Override input_size if specified
        if image_size is not None:
            config["input_size"] = image_size
            
        # Remove patch_size from config as it's set by the model variant
        config.pop("patch_size", None)
        
        model = DiT_FM_models[dit_model](**config)
    else:
        # Original UNet logic
        assert (
            architechture in MODEL_CONFIGS
        ), f"Model architecture {architechture} is missing its config."

        if is_discrete:
            if architechture + "_discrete" in MODEL_CONFIGS:
                config = MODEL_CONFIGS[architechture + "_discrete"]
            else:
                config = MODEL_CONFIGS[architechture]
            model = DiscreteUNetModel(
                vocab_size=257,
                **config,
            )
        else:
            model = UNetModel(**MODEL_CONFIGS[architechture])

    if use_ema:
        return EMA(model=model)
    else:
        return model
