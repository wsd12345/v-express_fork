import math
import time

import torch
import cv2 as cv
import numpy as np
from numpy import ndarray
from yacs.config import CfgNode
from PIL import Image
from typing import Iterator, Tuple, List, Optional

from models.pipeline import MyVExpressPipeline
from third_part.V_Express.pipelines.utils import draw_kps_image

from models.mutual_self_attention import MyReferenceAttentionControl
from predict.load_models import LipModels
from process.get_frames import get_reference_images


def get_reference_features(cfg: CfgNode, face_analysis_net, croper, path_face, num_max_read: int = -1,
                           ) -> Iterator[Tuple[List[ndarray], List[Image.Image], List[int], List[torch.Tensor]]]:
    # path_save = Path(path_save)
    # path_save.mkdir(parents=True, exist_ok=True)

    # models = LipModels(cfg, device, dtype)
    # app = models.get("face_analysis")
    #
    # croper = models.get("croper")

    image_iter = get_reference_images(cfg, croper, path_face, num_max_read)
    for i, (full_frames, full_faces, box) in enumerate(image_iter):
        kps_sequence = []
        for j, frame in enumerate(full_faces):  # todo  wsd_
            faces = face_analysis_net.get(cv.cvtColor(np.array(frame), cv.COLOR_RGB2BGR))
            assert len(faces) == 1, (f'There are {len(faces)} faces in the {(i - 1) * len(full_frames) + j}'
                                     f'-th frame. Only one face is supported.')

            kps = faces[0].kps[:3]
            kps_sequence.append(kps)
        yield full_frames, full_faces, box, kps_sequence


def get_kps_image(kps_sequence, height, width
                  ) -> Optional[List[Image.Image]]:
    if kps_sequence:
        kps_images = []
        for kps in kps_sequence:
            kps_image = draw_kps_image(height, width, kps)
            kps_images.append(Image.fromarray(kps_image))
    else:
        kps_images = None
    return kps_images
