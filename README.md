

Black and White image to Colorization.

# Black-and-White-Image-Colorization-with-OpenCV-and-DeepLearning

This project focuses on converting grayscale (black & white) images into realistic colorized images using a deep learning–based colorization model integrated with OpenCV.
The system takes a monochrome image as input and predicts the most probable colors for different regions based on image features and semantic understanding learned during training.

Colorization is a challenging task because a grayscale pixel provides only intensity information. The model must predict color values (a/b channels) using context — objects, edges, shapes, textures, and semantics.

->Project Objectives

Automatically colorize black & white images with high visual quality

Use a pretrained deep neural network model for lab-space color prediction

Implement an end-to-end colorization pipeline using OpenCV

Demonstrate deep learning–powered visual enhancement for historical images, photography restoration, and artistic effects

-> Deep Learning Approach

The project uses a pretrained Caffe-based deep learning model for image colorization, integrated with OpenCV’s DNN module.

-> How It Works

Convert the input image from RGB → LAB color space

Extract the L channel (lightness) — this is the grayscale input

Pass the L channel through the deep neural network

The model predicts the a/b chrominance channels

Merge predicted ab channels with original L channel

Convert back to BGR/RGB to obtain the colorized image

-> Pretrained Model

The project uses:

colorization_deploy_v2.prototxt (model architecture)

colorization_release_v2.caffemodel (trained weights)

These were trained on over 1.3 million images from the ImageNet dataset to learn color distribution patterns.

The model is from Zhang, Isola & Efros, "Colorful Image Colorization" (ECCV 2016) — https://github.com/richzhang/colorization

# Setup

`Model/colorization_release_v2.caffemodel` is stored with Git LFS. After cloning, run `git lfs install && git lfs pull`, or the model will fail to load.

```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`opencv-python` is pinned below 5 because OpenCV 5 removed Caffe model support.

# Usage

Put images in `Images/`, then:

```bash
python colorization.py
```

For each image:
1. The full colorized image is saved to `Output/<name>_colorized.jpg`.
2. A window asks you to select a region: drag a box, press Enter/Space (or `c` to skip).
3. The result shows the full colorization (left) and only the selected region colorized on the grayscale image (right), saved as `Output/<name>_region.jpg`.
4. Press any key for the next image; `Esc` or `q` quits.

Headless (no display), one image:

```bash
python -c "
import cv2
from colorization import colorize_full_image
cv2.imwrite('output.jpg', colorize_full_image(cv2.imread('Images/img_0.jpg')))
"
```
