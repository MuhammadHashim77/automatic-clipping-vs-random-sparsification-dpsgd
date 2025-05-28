# UNET factory function
# This module provides a function to create a UNET model using the segmentation_models_pytorch library.
import segmentation_models_pytorch as smp

def build_unet(img_channels: int = 1, out_classes: int = 1):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=img_channels,
        classes=out_classes,
    )
