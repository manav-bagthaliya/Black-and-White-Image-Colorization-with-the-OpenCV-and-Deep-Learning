# Commands

All commands assume you're in the project root:

```bash
cd "Black-and-White-Image-Colorization-with-the-OpenCV-and-Deep-Learning"
```

## 0. Prerequisite: fetch Git LFS files

`Model/colorization_release_v2.caffemodel` and `Dataset/cifar-10-python.tar.gz` are tracked with Git LFS.
Without `git-lfs` installed, these files are just small text pointers and the model will fail to load.

```bash
sudo apt install -y git-lfs   # one-time, needs a password prompt
git lfs install
git lfs pull
```

## 1. Set up the environment

```bash
python3 -m venv colorization_env
source colorization_env/bin/activate      # Windows: colorization_env\Scripts\activate
pip install -r requirements.txt
```

## 2. Prepare the dataset (optional)

Generates 10 sample grayscale images in `Images/` from the CIFAR-10 dataset in `Dataset/cifar-10-batches-py/`.

```bash
python PrepareDataset.py
```

## 3. Run colorization

Reads every image in `Images/`, colorizes the full image and a user-selected ROI side by side.
Requires a display (uses OpenCV GUI windows — `cv2.selectROI`, `cv2.imshow`).

```bash
python colorization.py
```

- A window opens per image to select the ROI (drag a box, press Enter/Space; press `c` to cancel).
- The result window shows the full colorized image (left) and colorized ROI (right).
- Press any key to move to the next image; close all windows when done.

## 3a. Headless colorization (no ROI selection)

To colorize a single image without a display, use `colorize_full_image()` directly:

```bash
python -c "
import cv2
from colorization import colorize_full_image
img = cv2.imread('Images/img_0.jpg')
cv2.imwrite('output.jpg', colorize_full_image(img))
"
```

## 4. Deactivate the environment

```bash
deactivate
```

## Notes

- Model files (`Model/colorization_deploy_v2.prototxt`, `Model/colorization_release_v2.caffemodel`, `Model/pts_in_hull.npy`) must already be present — they are checked in via Git LFS (see step 0).
- Input images are read from `Images/` (not `images/`) and the CIFAR-10 dataset from `Dataset/` (not `dataset/`) — paths are case-sensitive on Linux/macOS.
- `opencv-python` **5.x removed Caffe model support** (`cv2.dnn.readNetFromCaffe` no longer exists), which this project's model requires. `requirements.txt` pins `opencv-python<5` for that reason — don't upgrade past 4.x unless the model is ported to ONNX/ TensorFlow.
- `colorization_env/` (a virtualenv) should never be committed to git — it's now gitignored and untracked; recreate it locally with the step 1 commands instead of relying on a checked-in copy.
