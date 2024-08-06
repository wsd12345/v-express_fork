import math
import os
import cv2 as cv
import logging

import numpy as np
from PIL import Image

from numpy import ndarray
from typing import List, Tuple, Iterator

from yacs.config import CfgNode

from utils.get_pbar import Pbar

logger = logging.getLogger(__name__)

__full_width = None
__full_height = None
__box = None


def get_crop_image(croper, full_frames, full_width, full_height, face_width, box=None):
    full_frames = [cv.resize(i, (full_width, full_height)) for i in full_frames]
    full_frames_rgb = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in full_frames]

    if box is None:
        # face detection & cropping, cropping the first frame as the style of FFHQ
        full_faces_rgb, crop, quad, box = croper.crop(full_frames_rgb, xsize=512, out_size=face_width)
    else:
        full_faces_rgb = croper.update(full_frames_rgb, box)

    full_faces = [Image.fromarray(frame) for frame in full_faces_rgb]
    return full_frames, full_faces, box


def __yield(full_frames, croper, face_width, pbar):
    global __full_width, __full_height, __box
    if __box is None:
        h, w = full_frames[0].shape[:2]
        __full_width = w if __full_width == -1 else __full_width
        __full_height = h if __full_height == -1 else __full_height
    res = get_crop_image(croper, full_frames, __full_width, __full_height, face_width, box=__box)
    __box = res[2]
    pbar({"num": len(full_frames)}, state="add")
    return res


def get_frames(path_face: str, crop: List[int],
               croper,
               num_max_cache: int = 16,
               num_max_read: int = -1,
               verbose: bool = False, step: int = 0,
               full_width: int = -1, full_height: int = -1,
               face_width: int = -1, face_height: int = -1,
               ) -> Iterator[Tuple[List[ndarray], List[Image.Image], List[int]]]:
    full_frames: List[ndarray]
    frames_face: List[Image]
    global __full_width, __full_height, __box
    __full_width, __full_height = full_width, full_height

    assert face_height == face_width, "face_height != face_width"
    is_file = os.path.isfile(path_face)
    is_image = is_file and path_face.split('.')[-1] in ('jpg', 'png', 'jpeg')

    pbar = Pbar(verbose=verbose, desc=f"[Step {step}], data preprocessing")

    if not is_file:
        raise ValueError('--face argument must be a valid path to video/image file')

    elif is_image:
        full_frames = [cv.imread(path_face)]
        yield __yield(full_frames, croper, face_width, pbar)

    else:
        bs = math.ceil(num_max_read / num_max_cache) if num_max_read > 0 else float("inf")

        video_stream = cv.VideoCapture(path_face)

        full_frames = []
        y1, y2, x1, x2 = crop

        while True:

            still_reading, frame = video_stream.read()
            if not still_reading or bs <= 0:
                video_stream.release()
                yield __yield(full_frames, croper, face_width, pbar)
                break
            if len(full_frames) >= num_max_cache:
                yield __yield(full_frames, croper, face_width, pbar)
                bs -= 1
                full_frames = []

            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)


def get_reference_images(cfg: CfgNode, croper, path_face: str, num_max_read: int = 1,
                         ) -> Iterator[Tuple[List[ndarray], List[Image.Image], List[int]]]:
    frames_iter = get_frames(
        path_face, croper=croper, crop=[0, -1, 0, -1],
        num_max_cache=cfg.MAX_NUM_CACHE, num_max_read=num_max_read,
        face_width=cfg.WIDTH_FACE, face_height=cfg.HEIGHT_FACE,
    )
    for full_frames, full_faces, box in frames_iter:
        full_faces = [i.resize((cfg.WIDTH_NET, cfg.HEIGHT_NET)) for i in full_faces]
        yield full_frames, full_faces, box


class VideoIter:

    def __init__(self,
                 path_face: str):
        is_file = os.path.isfile(path_face)
        is_image = is_file and path_face.split('.')[-1] in ('jpg', 'png', 'jpeg')

        self.path_face = path_face
        if not is_file:
            raise ValueError('path_face must be a valid path to video/image file')
        elif is_image:
            self.next = self.__next_image
        else:
            self.next = self.__next_video

    def __iter__(self):
        return self.next()

    def __next__(self):

        return self.next()

    def __next_image(self):
        image = np.array(Image.open(self.path_face))[:, :, ::-1]
        while True:
            yield image

    def __next_video(self):
        while True:
            i = 0
            video_stream = cv.VideoCapture(self.path_face)
            while True:
                i += 1
                if i > 240:
                    break
                still_reading, frame = video_stream.read()
                if not still_reading:
                    break
                yield frame
            video_stream.release()


if __name__ == '__main__':

    obj = VideoIter(r"").__next__()
    for i, v in enumerate(obj):
        if i > 10:
            break
        print(v.shape)
    for i in range(10):
        next(obj)
