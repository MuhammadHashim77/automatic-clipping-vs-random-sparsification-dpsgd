# UNET factory function
# This module provides a function to create a UNET model using the segmentation_models_pytorch library.
import segmentation_models_pytorch as smp
import torch.nn as nn

def replace_bn_with_gn(model: nn.Module, num_groups: int = 32) -> nn.Module:
    """
    Replace all BatchNorm layers with GroupNorm layers.
    
    Args:
        model: The model to modify
        num_groups: Number of groups for GroupNorm
        
    Returns:
        Modified model with GroupNorm layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Calculate number of channels for the GroupNorm layer
            num_channels = module.num_features
            # Create new GroupNorm layer
            new_layer = nn.GroupNorm(
                num_groups=min(num_groups, num_channels),  # Ensure num_groups <= num_channels
                num_channels=num_channels,
                eps=module.eps,
                affine=module.affine
            )
            # Copy parameters if they exist
            if module.affine:
                new_layer.weight.data = module.weight.data
                new_layer.bias.data = module.bias.data
            # Replace the BatchNorm layer
            setattr(model, name, new_layer)
        else:
            # Recursively apply to child modules
            replace_bn_with_gn(module, num_groups)
    return model

def build_unet(img_channels: int = 1, out_classes: int = 1, num_groups: int = 32):
    """
    Build a U-Net model with GroupNorm instead of BatchNorm.
    
    Args:
        img_channels: Number of input channels
        out_classes: Number of output classes
        num_groups: Number of groups for GroupNorm layers
        
    Returns:
        U-Net model with GroupNorm
    """
    # Create base U-Net model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=img_channels,
        classes=out_classes,
    )
    
    # Replace BatchNorm with GroupNorm
    model = replace_bn_with_gn(model, num_groups)
    
    return model
