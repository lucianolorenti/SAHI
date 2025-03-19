#  Slicing Aided Hyper Inference (SAHI)
Simple and dependency free implmentation (just numpy and PIL) of SAHI

as presented in 

[Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://arxiv.org/abs/2202.06934) 



## Instalation
It°s easier just copy the __init__.py file to your repo
but 
```bash
pip install [-e] SAHI/
```
## Example
[See the notebook](notebook/example.ipynb)

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

You can watch me struggling implement this this [here](https://youtu.be/ZZks0ezRuPc)
