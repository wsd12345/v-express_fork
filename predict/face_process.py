import utils.insert_path
import os
import logging
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, NoReturn, Dict, Any, Optional, Union, Tuple, Iterator
from yacs.config import CfgNode
from functools import reduce
from operator import add
from constant import BASE_PATH
from models.pipeline import ReferencePipeline
from process.get_reference import get_kps_image, get_reference_features
from utils.get_mask import get_mask
from utils.save import MyPickle

logger = logging.getLogger(__name__)

model_save_feature_name = (
    "image_mask",
    "kps_feature",
    "reference_control_writer_weight"

)

directory_structure = """
- cache
    - 0
        - image_mask_0
        - kps_feature_0
        - reference_control_writer_weight_0
    - 1
    - 2
"""


def save_data(cfg: CfgNode,
              path_face: str, save_path: str,
              croper,
              vae,
              face_analysis_net,
              reference_net,
              v_kps_guider_net,
              num_max_read: int = -1,
              postfix: str = "bin",
              ):
    save_box = True

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    height, width = cfg.HEIGHT_NET, cfg.WIDTH_NET
    guidance_scale = cfg.DIFFUSION.GUIDANCE_SCALE
    context_frames = cfg.DIFFUSION.CONTEXT_FRAMES

    pipline = ReferencePipeline(
        vae=vae, reference_net=reference_net, v_kps_guider=v_kps_guider_net,
        denoising_unet=None, audio_encoder=None, audio_processor=None, audio_projection=None,
        scheduler=None)

    reference_iter = get_reference_features(cfg, face_analysis_net, croper, path_face, num_max_read)
    for i, (full_frames, full_faces, box, kps_sequence) in enumerate(reference_iter):
        kps_images = get_kps_image(kps_sequence, height, width)
        reference_control_writer_weight, kps_feature = pipline(
            full_faces,
            kps_images,
            width,
            height,
            context_frames,
            guidance_scale)

        image_mask = torch.from_numpy(np.stack([get_mask(img) for img in full_frames]))
        (save_path / f"{i}").mkdir(parents=True, exist_ok=True)
        torch.save(image_mask, save_path / f"{i}" / f"image_mask_{i}.{postfix}")
        torch.save(kps_feature, save_path / f"{i}" / f"kps_feature_{i}.{postfix}")
        torch.save(reference_control_writer_weight,
                   save_path / f"{i}" / f"reference_control_writer_weight_{i}.{postfix}")
        if save_box:
            torch.save(box,
                       save_path / f"box.{postfix}")
            save_box = False


def read_data(base_dir: Union[str, Path], names: List[str], index: int, postfix: str = "bin") -> Dict[str, Any]:
    base_dir = Path(base_dir) / f"{index}"
    res = {name: torch.load(base_dir / f"{name}_{index}.{postfix}") for name in names}

    return res


class FaceFeatureBase:
    def __init__(self,
                 base_dir: str,
                 context_frames: int,
                 context_overlap: int,
                 names: Optional[List[str]] = None,
                 postfix: str = "bin"):
        self.base_dir = Path(base_dir)
        self.names = names if names else model_save_feature_name
        self.postfix = postfix
        self.n = list([i for i in self.base_dir.glob("*") if i.is_dir()]).__len__()
        self.pre = {}
        self.indexes = list(range(self.n))
        self.context_frames = context_frames
        self.context_overlap = context_overlap
        self.bs = context_frames - context_overlap
        self.pad = 0

    def __iter__(self):
        return next(self)

    def __next__(self):
        raise NotImplemented

    def __getitem__(self, item) -> Dict[str, Any]:
        i = item % self.n
        return read_data(self.base_dir, self.names, i, self.postfix)

    def __len__(self):
        return len(self.indexes)


class FaceIterOne(FaceFeatureBase):
    def __init__(self,
                 base_dir: str,
                 context_frames: int,
                 context_overlap: int,
                 names: Optional[List[str]] = None,
                 postfix: str = "bin"):
        super().__init__(base_dir, context_frames, context_overlap, names, postfix)
        self.index = 0

    def set_index(self, x):
        if not 0 <= x < len(self.names):
            print(f"0<= x < {len(self.names)}, but x = {x}")
            return
        self.index = x

    def __next__(self):
        i = -1
        while True:
            i += 1
            masks = self[i][self.names[0]]
            for m in masks:
                yield m.type(torch.float32).numpy()


class FaceFeature(FaceFeatureBase):
    def __init__(self,
                 base_dir: str,
                 context_frames: int,
                 context_overlap: int,
                 names: Optional[List[str]] = None,
                 postfix: str = "bin"):
        super().__init__(base_dir, context_frames, context_overlap, names, postfix)
        if names and "reference_control_writer_weight" not in names:
            raise ValueError("reference_control_writer_weight not in names!")

    @property
    def pre_length(self):
        return self.__calc_bs(self.pre)

    def set_indexes(self, indexes: List[int], pad: int):
        self.indexes = indexes
        self.pad = pad - self.context_overlap

    def __calc_bs(self, x):
        if len(x):
            return x["reference_control_writer_weight"][0][0].shape[0]
        else:
            return 0

    def __split(self, x: Dict, idx):
        now, pre = {}, {}
        for k, v in x.items():
            if k == "image_mask":
                now[k] = v[:idx]
                pre[k] = v[idx:]
            elif k == "kps_feature":
                now[k] = v[:, :, :idx]
                pre[k] = v[:, :, idx:]
            elif k == "reference_control_writer_weight":
                now[k] = [[i[0][:idx]] for i in v]
                pre[k] = [[i[0][idx:]] for i in v]
        return now, pre

    def __add(self, data: List[Dict[str, Union[torch.Tensor, List]]]):
        res = {k: [v] for k, v in data[0].items()}
        for sub in data[1:]:
            for k, v in sub.items():
                res[k].append(v)
        for k, v in res.items():

            if k == "image_mask":
                res[k] = torch.concatenate(v, dim=0)

            elif k == "kps_feature":
                res[k] = torch.concatenate(v, dim=2)

            elif k == "reference_control_writer_weight":
                res[k] = [[torch.concatenate(reduce(add, [s_v[j] for s_v in v]), dim=0)]
                          for j in range(len(v[0]))]
        return res

    def __iter__(self):
        return next(self)

    def __next__(self):
        pre = self.pre
        now = []
        res = []
        i = 0
        while i < len(self):
            res.clear()
            pre_n = self.__calc_bs(pre)
            while pre_n < self.context_frames:
                if pre: res.append(pre)
                pre = self[self.indexes[i]]
                pre_n += self.__calc_bs(pre)
                i += 1
            now, pre = self.__split(pre, self.context_frames - pre_n)
            res.append(now)
            now = self.__add(res)
            yield now
            now, right = self.__split(now, self.bs)
            pre = self.__add([right, pre])

        self.pre = self.__add([self.__split(now, -self.pad)[1], pre])
        yield None

    def __getitem__(self, item) -> Dict[str, Any]:
        i = item % self.n
        return read_data(self.base_dir, self.names, i, self.postfix)

    def __len__(self):
        return len(self.indexes)


class __Test(FaceFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = 0

    def __next__(self):
        pre = self.pre
        now = []
        res = []
        i = 0
        while i < len(self):
            res.clear()
            pre_n = self.__calc_bs(pre)
            while pre_n < self.context_frames:
                if pre: res.append(pre)
                pre = self[self.indexes[i]]
                pre_n += self.__calc_bs(pre)
                i += 1
            now, pre = self.__split(pre, self.context_frames - pre_n)
            res.append(now)
            now = self.__add(res)
            yield now
            now, right = self.__split(now, self.bs)
            pre = self.__add([right, pre])

        self.pre = self.__add([self.__split(now, -self.pad)[1], pre])
        yield None

    def __getitem__(self, item):
        pre = self.i
        hed = self.i + self.context_frames - self.context_overlap
        self.i = hed
        return {"index": list(range(pre, hed))
                }

    def __calc_bs(self, x, *args, **kwargs):
        if len(x):
            return len(x["index"])
        else:
            return 0

    def __split(self, x: Dict, idx):
        now, pre = {}, {}
        for k, v in x.items():
            now[k] = v[:idx]
            pre[k] = v[idx:]

        return now, pre

    def __add(self, data: List[Dict[str, Union[torch.Tensor, List]]]):
        res = {k: [v] for k, v in data[0].items()}
        for sub in data[1:]:
            for k, v in sub.items():
                res[k].append(v)
        for k, v in res.items():
            res[k] = reduce(add, v)
        return res


if __name__ == '__main__':

    obj = __Test(base_dir=r".", context_frames=12, context_overlap=4)
    obj.set_indexes(list(range(6)), 7)
    for WSD in enumerate(obj):
        print(WSD)
    print(obj.pre)
    obj.set_indexes(list(range(6, 11)), 7)
    for WSD in enumerate(obj):
        print(WSD)
    print(obj.pre)
