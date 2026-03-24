This subfolder (with its own environment) was used to run the experiments in the following sections:

- 5.1. Objective 1: computational efficiency
- B.1. Not an approximation of full attention


## Environment setup with uv
```bash
# ensure uv is installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# ensure ninja is installed
sudo apt install ninja-build

uv init
uv add torch torchvision torchaudio pykeops matplotlib tqdm psutil
```

```bash
# only for benchmarking against FlashAttention
export CXX=g++
export CC=gcc
export MAX_JOBS=20
export CMAKE_BUILD_PARALLEL_LEVEL=20
export NINJA_NUM_JOBS=20
export BUILD_NINJA_PARALLEL=1
uv add flash-attn==2.6.3 --no-build-isolation   # this can take +- 30 mins

# to remove all uv packages: uv pip sync --allow-empty-requirements <(echo "")
```


## Navigating the codebase

We highlight some important files and folders in the codebase.

| File/Folder                | Description                                      |
|----------------------------|--------------------------------------------------|
| `methods/`                 | Contains various methods used in the experiments, including:<ul> <li> `naive_sparse.py` </li> <li> `symbolic_sparse.py` </li> <li> `full.py` </li> <li> `faiss.py` </li> <li> `annoy.py` </li> <li> `flash_attn.py` </li> <li> `post_processing.py` </li></ul> |
| `src/`                     | Contains the source code for the PyTorch C++ extension that we implemented for the Ball Tree Search algorithm.                     |
| `plotting.ipynb`           | Jupyter Notebook used for plotting results.           |
| `profiling-experiment.py`  | The main script for running the efficiency comparison experiments from 6.2 and 6.3.                |
| `approximation-experiment.py` | The main script for running the approximation quantification experiment from 6.7.         |


## Reproducing efficiency experiments

Inside the subfolder `profiling-experiment`, run:

```
# full attention
uv run profiling-experiment --method full

# k-MIP attention with naive brute-force search
uv run profiling-experiment --method naive-sparse

# k-MIP attention with symbolic matrices
uv run profiling-experiment --method sym
```

`profiling-experiment.py` has the following further arguments: `B, H, kq_dim, val_dim, k, do_backward, require_grad`

In our paper, we ran this experiment on a single 40GB A100 GPU.

Note: For running the approximation experiment, one would need the saved tokens in the GraphGPS repository.


<!-- Optional:
```
pip install ipykernel
pip install matplotlib
``` -->