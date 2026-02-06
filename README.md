# Fashion-MNIST Classification (PA3)

This is an end-to-end classification model trained on the  Fashion-MNIST dataset of clothing pictures using two PyTorch models: a fully-connected feedforward network (FFN) and a convolutional neural network (CNN).

This repository contains code and plots for training, evaluating, and visualizing models on the Fashion-MNIST dataset.

Suggested repository names (pick one):

- fashion-mnist-project
- pa3-fashion-classification
- fashion-cnn-ffn
- cloth-classifier-pytorch
- fashionmnist-visualizations

Contents
- `fashionmnist.py`: end-to-end training and plotting script (instantiates `FF_Net` and `Conv_Net`).
- `ffn.py`: feedforward network implementation (complete this file if not already).
- `cnn.py`: convolutional network implementation (complete this file if not already).
- `kernel_vis.py`: kernel/feature-visualization utilities.
- `requirements.txt`: Python dependencies.
- `PA3_splots.html`: simple display page with generated plot images.

Usage
1. Install requirements (recommended in a venv):

```bash
python3 -m pip install -r requirements.txt
```

2. Run training and plotting (this will download Fashion-MNIST and write several PNGs):

```bash
python3 fashionmnist.py
```

Files the script will produce (if training and plotting run successfully):
- `ffn.pth`, `cnn.pth` (model weights saved)
- `ffn_examples.png`, `cnn_examples.png` (one correct + one incorrect example each)
- `training_losses_ffn.png`, `training_losses_cnn.png` (training loss plots)
- kernel visualizations (if implemented in `kernel_vis.py`) e.g. `kernels.png`, `features.png`

Create the single PDF required for submission (Gradescope)

If you have ImageMagick installed you can combine the PNGs into one PDF:

```bash
convert ffn_examples.png cnn_examples.png training_losses_ffn.png training_losses_cnn.png kernels.png features.png PA3_splots.pdf
```

Or use Python to create a PDF (Pillow):

```bash
python3 - <<'PY'
from PIL import Image
imgs = [Image.open(x).convert('RGB') for x in ['ffn_examples.png','cnn_examples.png','training_losses_ffn.png','training_losses_cnn.png'] if __import__('os').path.exists(x)]
if imgs:
    imgs[0].save('PA3_splots.pdf', save_all=True, append_images=imgs[1:])
PY
```

Notes
- The autograder expects the transforms in `fashionmnist.py` to remain unchanged.
- Ensure `ffn.py` and `cnn.py` implement `FF_Net` and `Conv_Net` classes and return logits suitable for `nn.CrossEntropyLoss`.

Display page
- See `PA3_splots.html` for a lightweight in-repo gallery of the generated plots and examples.

If you want, I can also:
- generate a combined `PA3_splots.pdf` for you (I can run the training locally if you want me to train here), or
- open a GitHub Pages-ready `index.html` and push a branch ready for publishing.
