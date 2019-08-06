# Chest X-Ray Analysis

## How I setup my environment
- Windows 10
- Anaconda 3
- CUDA 10.0 & CuDNN
- `nb_conda_kernel` (to use Jupyter Notebook in a virtual environment)
- TensorFlow > 1.14 (There are some bugs in TF 1.13 and lower)

```bash
$ conda create -n chestxray python=3.6 ipykernel
$ conda activate chestxray
$ pip install tensorflow-gpu
$ pip install matplotlib
```

## Segmentation
### Datasets
See [data/README.md](data/README.md) for the details
