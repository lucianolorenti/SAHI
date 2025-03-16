#  Slicing Aided Hyper Inference (SAHI)

Simple and dependency free implmentation (just numpy and PIL) of SAHI

## Example
See the notebook []


```python
from sahi import SAHIConfig, keras_cv_predict, sahi_predict
predict = partial(keras_cv_predict, pretrained_model)
predictions = sahi_predict(
    predict,
    np.array(img),
    SAHIConfig(
        patch_size=(64, 64),
        overlapping_prportion=(0.5, 0.5),
        model_image_size=(416, 416),
        iou_threshold=0.3
    ),
)
```