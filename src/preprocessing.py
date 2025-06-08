import numpy as np
import torch
import albumentations as A
from pathlib import Path
from typing import Tuple, Optional
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

class MedicalImagePreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        voi_lut: bool = True,
        augment: bool = True
    ):
        """
        Initialize the medical image preprocessor.
        
        Args:
            target_size: Target size for resizing images
            normalize: Whether to normalize pixel values
            clip_range: Range for clipping normalized values
            voi_lut: Whether to apply VOI LUT for DICOM images
            augment: Whether to apply augmentations
        """
        self.target_size = target_size
        self.normalize = normalize
        self.clip_range = clip_range
        self.voi_lut = voi_lut
        self.augment = augment
        
        # Define augmentations
        self.train_transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            ),
            A.RandomBrightnessContrast(0.05, 0.05, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(p=0.2),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
        ])

    def preprocess_dicom(self, path: str | Path) -> np.ndarray:
        """
        Preprocess a DICOM image.
        
        Args:
            path: Path to the DICOM file
            
        Returns:
            Preprocessed image as numpy array
        """
        ds = pydicom.dcmread(str(path))
        
        # Convert to float32
        img = ds.pixel_array.astype(np.float32)
        
        # Apply modality LUT
        img = apply_modality_lut(img, ds)
        
        # Apply VOI LUT if requested
        if self.voi_lut:
            img = apply_voi_lut(img, ds)
        
        # Normalize to 0-1
        img -= img.min()
        img /= max(img.max(), 1e-3)
        
        # Invert if MONOCHROME1
        if ds.PhotometricInterpretation == "MONOCHROME1":
            img = 1.0 - img
            
        return img

    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocess a binary mask.
        
        Args:
            mask: Binary mask as numpy array
            
        Returns:
            Preprocessed mask
        """
        # Ensure binary values
        mask = (mask > 0).astype(np.float32)
        return mask

    def apply_transforms(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply transformations to image and mask.
        
        Args:
            image: Input image
            mask: Optional mask
            is_training: Whether to apply training augmentations
            
        Returns:
            Transformed image and mask
        """
        # Add channel dimension if needed
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
            
        # Apply transformations
        transform = self.train_transform if (is_training and self.augment) else self.val_transform
        if mask is not None:
            transformed = transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        else:
            transformed = transform(image=image)
            image = transformed['image']
            
        # Normalize if requested
        if self.normalize:
            image = (image - 0.5) * 2  # Scale to [-1, 1]
            image = np.clip(image, self.clip_range[0], self.clip_range[1])
            
        return image, mask

    def to_tensor(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert numpy arrays to PyTorch tensors.
        
        Args:
            image: Image array
            mask: Optional mask array
            
        Returns:
            Image and mask tensors
        """
        # Convert image to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Convert mask to tensor if provided
        mask_tensor = None
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            
        return image_tensor, mask_tensor

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            image: Input image
            mask: Optional mask
            is_training: Whether to apply training augmentations
            
        Returns:
            Preprocessed image and mask tensors
        """
        # Preprocess image
        if isinstance(image, (str, Path)):
            image = self.preprocess_dicom(image)
            
        # Preprocess mask if provided
        if mask is not None:
            mask = self.preprocess_mask(mask)
            
        # Apply transformations
        image, mask = self.apply_transforms(image, mask, is_training)
        
        # Convert to tensors
        return self.to_tensor(image, mask) 