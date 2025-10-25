# Neural Style Transfer Project 🎨

## Overview
This project implements Neural Style Transfer (NST) utilizing a pre-trained VGG19 convolutional neural network.  
The objective is to generate a new image that preserves the structural content of a content image while transferring the artistic characteristics of a style image.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Truong-Anh-Kiet/neural-style-transfer.git
   cd neural-style-transfer
   ```
2. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
*Note:*
The program automatically detects GPU availability (torch.cuda.is_available()).
If a CUDA-enabled GPU is available, computation will run on GPU for faster optimization;
otherwise, it will fall back to CPU mode seamlessly.

## Execution
To perform NST, place content images in the `contents/` directory and a selected `style.jpg`.  
Then execute:
```bash
python nst.py
```
Stylized images will be automatically generated in the `output/` folder.

## Experimental Settings
- **Backbone Model:** VGG19 (pre-trained on ImageNet, frozen parameters)  
- **Content Layer:** `conv4_2`  
- **Style Layers:** `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`  
- **Optimizer:** L-BFGS  
- **Hyperparameters:** α = 1, β = 10000, total steps = 300, image_size = 512
- **Framework:** PyTorch 2.0+

## Citation
If you use or refer to this project in academic or research work, please cite as:
```
@misc{neuralstyle2025,
  title  = {Neural Style Transfer using VGG19},
  author = {Truong Anh Kiet},
  year   = {2025},
  note   = {Mini Project, FPT University}
}
```

## Acknowledgment
This implementation is inspired by the seminal paper *"A Neural Algorithm of Artistic Style"*  
by **Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge (2015)**.  
The work follows the foundational concept of optimizing image representations through convolutional feature maps.

## License
This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with proper attribution.
