"""
Slicing Aided Hyper Inference (SAHI): First, the original query image $I$ is sliced into $l$ number of $M \times N$ overlapping patches $P_1^I, P_2^I, ..., P_l^I$. Then, each patch
is resized while preserving the aspect ratio. After that, object detection forward pass is applied independently to each overlapping patch. An optional full-inference (FI) using the
original image can be applied to detect larger objects.

Finally, the overlapping prediction results and, if used, FI results are merged back into to original size using NMS. During NMS, boxes having higher Intersection over Union (IoU) ratios than a predefined matching threshold $T_m$ are matched and for each match, detections having detection probability than lower than $T_d$ are remov"
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import PIL
from PIL import Image

logger = logging.getLogger(__name__)
SAHI_HOME = Path(__file__).parent.parent


@dataclass
class SAHIConfig:
    patch_size: Tuple[int, int] = (300, 300)
    overlapping_prportion: Tuple[float, float] = (0.5, 0.5)
    model_image_size: Tuple[int, int] = (300, 300)
    use_whole_image: bool = True
    iou_threshold: float = 0.5
    batch_size: Optional[int] = None


@dataclass
class ModelPredictions:
    num_detections: int
    boxes: np.ndarray
    confidence: np.ndarray
    classes: np.ndarray


def resize_and_pad(
    img: np.ndarray, target_size: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """Reize the image while preserving the aspect ratio and padding the image.

    Args:
        img (np.ndarray): Image
        target_size (Tuple[int, int]): Target size

    Returns:
        Tuple[np.ndarray, Tuple[int, int], float]: Resized image, padding used, scaling factor
    """
    orign_nrows, origin_ncols = img.shape[:2]
    target_ncols, target_nrows = target_size
    scaling_factor = min(target_nrows / orign_nrows, target_ncols / origin_ncols)
    concrete_target_size = (
        int(origin_ncols * scaling_factor),
        int(orign_nrows * scaling_factor),
    )
    image = PIL.Image.fromarray(img).resize(concrete_target_size)
    result = PIL.Image.new(image.mode, target_size, (0, 0, 0))
    top = (target_size[1] - concrete_target_size[1]) // 2
    left = (target_size[0] - concrete_target_size[0]) // 2
    result.paste(image, (left, top))
    return np.array(result), (left, top), scaling_factor


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

        self.target_img, (left, top), scaling_factor = resize_and_pad(
            self.img, target_size
        )

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
    stride_x = int(patch_size[0] * (1 - overlapping_prportion[0]))
    stride_y = int(patch_size[1] * (1 - overlapping_prportion[1]))
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


def prepare_patches(
    patches: List[ImagePatch], img: np.ndarray, config: SAHIConfig
) -> List[ImagePatch]:
    """Prepares the patches for the model.

    Resize the patches to the required model image size preserving the aspect ratio.
    Also adds the whole image if required.

    Args:
        patches (List[ImagePatch]): List of image patches
        img (np.ndarray): Original image
        config (SAHIConfig): SAHI Configuration

    Returns:
        List[ImagePatch]: List of image patches prepared for the model
    """
    for patch in patches:
        patch.rescaling_preserving_aspect_ratio(target_size=config.model_image_size)
    if config.use_whole_image:
        patches.append(
            ImagePatch(
                img=img, starting_row=0, starting_col=0
            ).rescaling_preserving_aspect_ratio(target_size=config.model_image_size)
        )
    return patches


def map_bbs_to_original_image(bbs: ModelPredictions, patches: List[ImagePatch]) -> dict:
    """Maps the predicted bounding boxes from patches to the original image coordinates.

    Args:
        bbs (dict): Dictionary with four keys: num_detections, boxes, confidence, classes
        patches (List[ImagePatch]): List of image patches used to predict

    Returns:
        dict: Mapped bounding boxes. A dictionary with four keys: num_detections, boxes, confidence, classes
            The resulting bounding boxes are the ones with a prediction.
    """
    n_patches = len(patches)
    num_detections = bbs.num_detections
    boxes = bbs.boxes
    confidence = bbs.confidence
    classes = bbs.classes

    final_boxes = []
    final_confidence = []
    final_classes = []
    final_number_of_detections = 0
    for i in range(n_patches):
        num_detections_patch = num_detections[i]
        final_number_of_detections += num_detections_patch
        patch_boxes = patches[i].map_bbs_to_original_image_coordinates(
            boxes[i, :num_detections_patch, :]
        )
        patch_confidences = confidence[i, :num_detections_patch]
        patch_classes = classes[i, :num_detections_patch]

        final_boxes.append(patch_boxes)
        final_confidence.append(patch_confidences)
        final_classes.append(patch_classes)

    return ModelPredictions(
        num_detections=final_number_of_detections,
        boxes=np.concatenate(final_boxes).reshape(-1, 4),
        confidence=np.concatenate(final_confidence).reshape(-1),
        classes=np.concatenate(final_classes).reshape(-1),
    )


def iou_of_boxes(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calculates the Intersection over Union (IoU) of a box with a set of boxes.

    Args:
        box (np.ndarray): Box in format xywh
        boxes (np.ndarray): Set of boxes in format xywh

    Returns:
        np.ndarray: IoU of the box with the set of boxes
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - intersection
    eps = 1e-6
    return intersection / (union + eps)


def non_maximum_suppression(
    boxes: np.ndarray, confidence: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Applies non-maximum suppression to the boxes.

    Args:
        boxes (np.ndarray): Bounding boxes
        confidence (np.array): Confidence of the bounding boxes
        iou_threshold (float): IoU threshold

    Returns:
        np.ndarray: Indices of the boxes after non-maximum suppression
    """
    indices = confidence.argsort()[::-1]
    out = []
    while len(indices) > 0:
        i = indices[0]
        out.append(i)
        remaining_indices = indices[1:]
        if len(remaining_indices) == 0:
            break
        iou = iou_of_boxes(boxes[i], boxes[remaining_indices])
        indices = np.delete(remaining_indices, np.where(iou > iou_threshold))
    return np.array(out)


def multiclass_non_maximum_suppression(
    bounding_boxes: ModelPredictions, iou_threshold: float
) -> ModelPredictions:
    """Applies non-maximum suppression to the bounding boxes per class.

    Args:
        bounding_boxes (ModelPredictions): Set of predictions
        iou_threshold (float): IoU threshold

    Returns:
        ModelPredictions: Predictions after non-maximum suppression
    """
    boxes = bounding_boxes.boxes
    confidence = bounding_boxes.confidence
    classes = bounding_boxes.classes

    out_boxes = []
    out_confidences = []
    out_classes = []
    num_detections = 0
    for class_ in np.unique(classes):
        class_indices = np.where(classes == class_)[0]
        class_boxes = boxes[class_indices]
        class_confidence = confidence[class_indices]

        indices = non_maximum_suppression(
            class_boxes,
            class_confidence,
            iou_threshold=iou_threshold,
        )
        out_boxes.append(class_boxes[indices])
        out_confidences.append(class_confidence[indices])
        out_classes.append(class_ * np.ones_like(class_confidence[indices]))
        num_detections += len(indices)

    return ModelPredictions(
        num_detections=num_detections,
        boxes=np.concatenate(out_boxes).reshape(-1, 4),
        confidence=np.concatenate(out_confidences).reshape(-1),
        classes=np.concatenate(out_classes).reshape(-1),
    )


def keras_cv_predict(model, images: np.ndarray, batch_size: int) -> ModelPredictions:
    """Predicts the bounding boxes of objects in the images using the model.

    Useful function for using it together with functools.partial to create a model_predict function
    to be used in sahi_predict.

    Args:
        model (Keras Model):
        image (np.ndarray): Image to predict
        batch_size (int): Batch size

    Returns:
        ModelPredictions: Predictions
    """
    model_raw_prediction = model.predict(images, batch_size=batch_size)
    return ModelPredictions(
        num_detections=model_raw_prediction["num_detections"],
        boxes=model_raw_prediction["boxes"],
        confidence=model_raw_prediction["confidence"],
        classes=model_raw_prediction["classes"],
    )


def sahi_predict(
    model_predict: Callable[[np.ndarray, int], ModelPredictions],
    img: np.ndarray,
    config: SAHIConfig,
):
    """
    Predicts the bounding boxes of objects in the image using the SAHI algorithm.
    """

    patches = split_image_in_windows(
        img,
        patch_size=config.patch_size,
        overlapping_prportion=config.overlapping_prportion,
    )

    patches = prepare_patches(patches, img, config)

    logger.debug(f"Number of patches: {len(patches)}")

    images_to_predict = np.concatenate(
        [np.expand_dims(patch.target_img, 0) for patch in patches]
    )
    batch_size = len(images_to_predict)
    if config.batch_size:
        batch_size = config.batch_size

    model_prediction = model_predict(images_to_predict, batch_size)

    model_prediction = map_bbs_to_original_image(model_prediction, patches)
    return multiclass_non_maximum_suppression(
        model_prediction, iou_threshold=config.iou_threshold
    )
