"""
Slicing Aided Hyper Inference (SAHI): First, the original query image $I$ is sliced into $l$ number of $M \times N$ overlapping patches $P_1^I, P_2^I, ..., P_l^I$. Then, each patch
is resized while preserving the aspect ratio. After that, object detection forward pass is applied independently to each overlapping patch. An optional full-inference (FI) using the
original image can be applied to detect larger objects.

Finally, the overlapping prediction results and, if used, FI results are merged back into to original size using NMS. During NMS, boxes having higher Intersection over Union (IoU) ratios than a predefined matching threshold $T_m$ are matched and for each match, detections having detection probability than lower than $T_d$ are remov"
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Tuple, List
import PIL
from PIL import Image
import logging 

logger = logging.getLogger(__name__)
SAHI_HOME = Path(__file__).parent.parent


class ImagePatch:
    def __init__(self, img: np.ndarray, starting_row: int, starting_col: int):
        self.img = img
        self.target_img = img
        self.starting_row = starting_row
        self.starting_col = starting_col
        self.padding_rescaled_left = 0
        self.padding_rescaled_top = 0
        self.scaling_factor = 1.0

    def rescaling_preserving_aspect_ratio(
        self, target_size: Tuple[int, int]
    ) -> "ImagePatch":
        """Rescales the image while preserving the aspect ratio.

        Args:
            img (np.ndarray): Image
            target_size (Tuple[int, int], optional): The required target size. (width, height)

        Returns:
            Tuple[PIL.Image, Tuple[int, int]]: Rescaled image and the padding used
        """

        orign_nrows, origin_ncols = self.img.shape[:2]
        target_ncols, target_nrows = target_size
        scaling_factor = min(target_nrows / orign_nrows, target_ncols / origin_ncols)
        concrete_target_size = (
            int(origin_ncols * scaling_factor),
            int(orign_nrows * scaling_factor),
        )
        image = PIL.Image.fromarray(self.img).resize(concrete_target_size)
        result = Image.new(image.mode, target_size, (0, 0, 0))
        top = (target_size[1] - concrete_target_size[1]) // 2
        left = (target_size[0] - concrete_target_size[0]) // 2
        result.paste(image, (left, top))
        self.target_img = np.array(result)
        self.padding_rescaled_left = left
        self.padding_rescaled_top = top
        self.scaling_factor = scaling_factor
        return self
    
    def map_bbs_to_original_image_coordinates(self, bbs: np.ndarray) -> np.ndarray:
        """Maps the bounding boxes to the original image coordinates.

        Args:
            bbs (np.ndarray): Bounding boxes in format xywh

        Returns:
            np.ndarray: Mapped bounding boxes
        """
        bbs[:, 0] -= self.padding_rescaled_left
        bbs[:, 1] -= self.padding_rescaled_top        
        bbs /= self.scaling_factor
        bbs[:, 0] += self.starting_col
        bbs[:, 1] += self.starting_row
        return bbs


def split_image_in_windows(
    img: np.ndarray,
    patch_size: Tuple[int, int] = (300, 300),
    overlapping_prportion: Tuple[float, float] = (0.5, 0.5),
) -> List[ImagePatch]:
    """Splits the image into overlapping patches.

    Args:
        img (np.ndarray): Original image
        patch_size (Tuple[int, int], optional): Size of each patch. Defaults to (300, 300).
        overlapping_prportion (Tuple[float, float], optional): Overlapping proportion. Defaults to (0.5, 0.5).

    Returns:
        List[ImagePatch]: List of images patches
    """

    def valid_proportion(v: float) -> bool:
        return v >= 0 and v < 1

    if not valid_proportion(overlapping_prportion[0]) or not valid_proportion(
        overlapping_prportion[1]
    ):
        raise ValueError("Overlapping proportion should be between 0 and 1.")
    patches = []
    h, w, _ = img.shape
    stride_x = int(patch_size[0] * (1-overlapping_prportion[0]))
    stride_y = int(patch_size[1] * (1-overlapping_prportion[1]))
    for i in range(0, h, stride_y):
        for j in range(0, w, stride_x):
            rows_start = i
            rows_end = min(i + patch_size[0], h)
            cols_start = j
            cols_end = min(j + patch_size[1], w)
            patch = img[rows_start:rows_end, cols_start:cols_end]
            patches.append(
                ImagePatch(img=patch, starting_row=rows_start, starting_col=cols_start)
            )
    return patches


def map_bb_to_original_image(
    bb: np.ndarray, patch: ImagePatch, original_image: Tuple[int, int]
) -> np.ndarray:
    """Maps the bounding box to the original image.

    Args:
        bb (np.ndarray): Bounding box in formta xywh
        patch (ImagePatch): Image patch
        original_image (Tuple[int, int]): Original image size

    Returns:
        np.ndarray: Mapped bounding box
    """
    h, w = original_image
    x, y, w, h = bb
    x += patch.starting_col
    y += patch.starting_row
    return np.array([x, y, w, h])


@dataclass
class SAHIConfig:
    patch_size: Tuple[int, int] = (300, 300)
    overlapping_prportion: Tuple[float, float] = (0.5, 0.5)
    model_image_size: Tuple[int, int] = (300, 300)


def sahi_predict(model, img: np.ndarray, config: SAHIConfig):
    """
    Predicts the bounding boxes of objects in the image using the SAHI algorithm.
    """
    ## Image + Patch coordinates wrt to the original image
    patches = split_image_in_windows(
        img,
        patch_size=config.patch_size,
        overlapping_prportion=config.overlapping_prportion,
    )
    logger.debug(f"Number of patches: {len(patches)}")
    for patch in patches:
        patch.rescaling_preserving_aspect_ratio(
            target_size=config.model_image_size
        )

    images_to_predict = np.concatenate([np.expand_dims(patch.target_img, 0) for patch in patches])
    bbs = model.predict(images_to_predict, batch_size=len(images_to_predict))

    n_patches = len(patches)
    num_detections = bbs['num_detections']
    boxes = bbs['boxes']
    confidence = bbs['confidence']
    classes = bbs['classes']
    for i in range(n_patches):        
        num_detections_patch = num_detections[i]
        boxes[i][:num_detections_patch] = patches[i].map_bbs_to_original_image_coordinates(boxes[i][:num_detections_patch])
        
    boxes = boxes.reshape(-1, 4)
    confidence = confidence.reshape(-1)
    classes = classes.reshape(-1)
        
    return {
        "num_detections": num_detections,
        "boxes": boxes,
        "confidence": confidence,
        "classes": classes
    }

    
    
    ## Map the bounding boxes back to the original coordinates
    bbs = map_bbs_to_original_image(bbs, patches)

    ## Merge the bounding boxes
    bbs = merge_bb_nms(bbs)
