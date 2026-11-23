import glob
import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")
IMAGE_DIR = os.path.join(BASE_DIR, "Images")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")

net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt"),
    os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel"),
)
pts = np.load(os.path.join(MODEL_DIR, "pts_in_hull.npy")).transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Max display size (width, height); images are scaled to fit, keeping aspect ratio.
MAX_WIDTH = 640
MAX_HEIGHT = 480


def colorize_full_image(img):
    """Colorize a BGR uint8 image and return a BGR uint8 image of the same size."""
    lab_img = cv2.cvtColor(img.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)

    L = cv2.split(cv2.resize(lab_img, (224, 224)))[0]
    L -= 50  # model expects mean-centred L

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    colorized = np.concatenate((lab_img[:, :, :1], ab), axis=2)
    colorized = np.clip(cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR), 0, 1)
    return (255 * colorized).astype("uint8")


SELECT_WINDOW = "Select Object to Colorize"
RESULT_WINDOW = "Image Colorization"


class WindowClosed(Exception):
    """User closed a window with its X button, or pressed Esc/q: stop processing images."""


def window_closed(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:  # Qt backend raises once the window has been destroyed
        return True


def select_roi(img):
    """Drag a box, Enter/Space confirms, c skips. Returns (x, y, w, h) or None; raises WindowClosed."""
    # Own loop instead of cv2.selectROI: selectROI re-shows its window every frame, so X can't close it.
    box = {}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            box.update(start=(x, y), end=(x, y), dragging=True)
        elif event == cv2.EVENT_MOUSEMOVE and box.get("dragging"):
            box["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and box.get("dragging"):
            box.update(end=(x, y), dragging=False)

    cv2.namedWindow(SELECT_WINDOW)
    cv2.setMouseCallback(SELECT_WINDOW, on_mouse)
    while True:
        view = img.copy()
        if "start" in box:
            cv2.rectangle(view, box["start"], box["end"], (0, 255, 0), 2)
        cv2.imshow(SELECT_WINDOW, view)
        key = cv2.waitKey(20)
        # Check before the next imshow, which would re-open a closed window.
        if window_closed(SELECT_WINDOW) or (key & 0xFF) in (27, ord("q")):
            raise WindowClosed
        if (key & 0xFF) in (13, 32, ord("c")):
            break

    cv2.destroyWindow(SELECT_WINDOW)
    if key & 0xFF == ord("c") or "start" not in box:
        return None
    (x1, y1), (x2, y2) = box["start"], box["end"]
    h_img, w_img = img.shape[:2]
    x1, x2 = sorted((min(max(x1, 0), w_img), min(max(x2, 0), w_img)))
    y1, y2 = sorted((min(max(y1, 0), h_img), min(max(y2, 0), h_img)))
    if x2 - x1 == 0 or y2 - y1 == 0:
        return None
    return x1, y1, x2 - x1, y2 - y1


def colorize_selected_region(img):
    """Let the user pick a region, colorize only that region. Returns (image, roi) or (None, None)."""
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    roi = select_roi(gray)
    if roi is None:
        return None, None
    x, y, w, h = roi
    return colorize_full_image(img[y:y + h, x:x + w]), roi


def fit(img, max_w=MAX_WIDTH, max_h=MAX_HEIGHT):
    """Downscale to fit within max_w x max_h, keeping aspect ratio."""
    scale = min(max_w / img.shape[1], max_h / img.shape[0], 1.0)
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))))


def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*")))
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        for img_path in image_paths:
            process_image(img_path)
    except WindowClosed:
        print("Window closed, stopping.")
    finally:
        cv2.destroyAllWindows()


def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        return

    img = fit(img)
    full = colorize_full_image(img)
    name = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_colorized.jpg"), full)

    region, roi = colorize_selected_region(img)
    if region is None:
        print(f"No ROI selected for {img_path}, showing full colorization only.")
        result = full
    else:
        # Region shown on the grayscale image so the selected object stands out.
        x, y, w, h = roi
        region_view = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        region_view[y:y + h, x:x + w] = region
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_region.jpg"), region_view)
        result = np.hstack((full, region_view))

    cv2.imshow(RESULT_WINDOW, result)
    # Poll instead of waitKey(0) so clicking X on the result window is noticed.
    while (key := cv2.waitKey(50)) == -1:
        if window_closed(RESULT_WINDOW):
            raise WindowClosed
    if (key & 0xFF) in (27, ord("q")):
        raise WindowClosed


if __name__ == "__main__":
    main()
