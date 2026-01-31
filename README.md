# Machine Learning Concepts With Code

A learning repo that implements **core ML / DL building blocks from scratch** (mostly **NumPy-only**), with **step-by-step notebooks** that explain the math, shapes, and backprop.

## Scope (current)

- Implement foundational components **without using PyTorch / torch** or high-level deep learning APIs.
- Keep code **readable and inspectable** (explicit intermediate variables > clever tricks).
- Provide **playground notebooks** that explain each module/architecture step-by-step.
- Include **small runnable tests / scripts** (CPU-friendly) to validate correctness.

## Contents

- **DNN/** — Dense layer, activations, losses, and optimizers (NumPy)
  - `Neuron.py`, `activation_functions.py`, `loss_functions.py`, `optimizers.py`
  - `playground.ipynb`

- **CNN/** — Convolutional building blocks (NumPy)
  - `layers.py`: `Layer_Conv2D_Im2Col`, `Layer_MaxPool2D`, `Layer_Flatten`
  - `playground.ipynb`

- **BatchNorm/** — Batch Normalization (NumPy)
  - `batchnorm.py` + `test_batchnorm.py`
  - `playground.ipynb`

- **LayerNorm/** — Layer Normalization (NumPy)
  - `layernorm.py` + `test_layernorm.py`
  - `playground.ipynb`

- **ResNet/** — ResNet-18 style network for MNIST **built from this repo’s CNN + DNN blocks** (NumPy)
  - `resnet18_numpy.py`, `mnist_data.py`, `train_mnist.py`, `test_resnet_smoke.py`
  - `playground.ipynb`

- **Attention/** — attention walkthrough notebook(s)

## Getting started

### Option A: Just read notebooks
1. Install Python (3.10+ recommended) and Jupyter.
2. Clone:
   ```bash
   git clone https://github.com/sssstf0rest/Machine-Learning-Concepts-With-Code.git
   cd Machine-Learning-Concepts-With-Code
   ```
3. Start Jupyter and open any `playground.ipynb`:
   ```bash
   jupyter notebook
   ```

### Option B: Run quick checks (recommended)
Create a virtual environment (recommended) and install NumPy:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip numpy
```

Then run:
```bash
python -m BatchNorm.test_batchnorm
python -m LayerNorm.test_layernorm
python -m ResNet.test_resnet_smoke
```

### Train ResNet on MNIST (fast CPU run)
```bash
python -m ResNet.train_mnist --epochs 1 --subset 5000 --batch-size 64 --lr 1e-3
```

Notes:
- Full MNIST training with a ResNet-18 style model in pure NumPy can be slow; use `--subset` for iteration.

## Design conventions

Across the repo, layers generally follow:
- `forward(inputs)` sets `.output`
- `backward(dvalues)` sets `.dinputs` and parameter gradients (e.g., `.dweights`, `.dbiases`) when applicable

This makes it easy to compose layers and reuse optimizers.

## Contributing / next steps

If you add new concepts:
- include a `playground.ipynb` explaining the math + structure
- add at least one small test / sanity script
- keep dependencies minimal

## License

MIT
