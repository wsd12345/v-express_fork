from typing import List

import cv2 as cv
import numpy as np
from PIL import Image


def merge_face2image(full, face, box, n=50):
    oy1, oy2, ox1, ox2 = box
    sy = oy2 - n

    res = full.astype(np.float32)
    res[oy1:sy, ox1:ox2] = face[:-n, :]

    rate = (np.arange(n) / n)[:, np.newaxis, np.newaxis]
    res[sy:oy2, ox1:ox2] *= rate
    res[sy:oy2, ox1:ox2] += face[-n:, :] * (1 - rate)
    return res.astype(np.uint8)


def merge_face2image(full: np.ndarray, face: np.ndarray, box: List[int], n: int = 50
                     ) -> np.ndarray:

    oy1, oy2, ox1, ox2 = box
    sy = oy2 - n

    # In order to ensure the unification of resize image
    face = np.array(Image.fromarray(face.astype(np.uint8)).resize(size=(ox2 - ox1, oy2 - oy1)))

    face = face.astype(np.float32)
    res = full.astype(np.float32)
    res[oy1:sy, ox1:ox2] = face[:-n, :]

    rate = (np.arange(n) / n)[:, np.newaxis, np.newaxis]
    res[sy:oy2, ox1:ox2] *= rate
    res[sy:oy2, ox1:ox2] += face[-n:, :] * (1 - rate)
    return res.astype(np.float32)


def get_mask(img: np.ndarray) -> np.ndarray:


    img_t = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_t, (35, 43, 46), (77, 255, 255)) + 1
    mask = mask[:, :, np.newaxis]

    return mask


def get_mask_image(img: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:

    return ((1 - mask) * background + mask * img ).astype(np.uint8)


if __name__ == '__main__':
    import cv2 as cv
    import matplotlib.pyplot as plt

    path_face = r""
    path_background = r""
    video_stream = cv.VideoCapture(path_face)

    still_reading, frame = video_stream.read()
    video_stream.release()
    frame_h, frame_w = frame.shape[:2]
    background = cv.imread(path_background)
    background = cv.resize(background, (frame_w, frame_h))

    new_pos = 200
    b_new = np.zeros_like(background)
    b_new[:frame_h - new_pos, ] = background[new_pos:, ]
    mask = get_mask(frame)
    a = get_mask_image(frame, mask, b_new)
    cv.imwrite(r"", b_new)
