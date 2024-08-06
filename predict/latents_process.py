from pathlib import Path
from typing import Tuple
from collections import deque
import torch

from models.pipeline import LipPipeline


class Latents:
    def __init__(self,
                 cfg,
                 base_dir: str,
                 lip_pipeline: LipPipeline,
                 batch_size,
                 num_channels_latents,
                 width,
                 height,
                 video_length,
                 dtype,
                 device,
                 generator,
                 ):

        self.cfg = cfg
        self.device = device
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path_format = str(self.base_dir / "{0}.bin")

        context_overlap = self.cfg.DIFFUSION.CONTEXT_OVERLAP
        bs = self.cfg.MAX_NUM_CACHE
        a, b = divmod((video_length - context_overlap), bs)
        self.length = a
        self.context_overlap = context_overlap
        if b != 0:
            raise ValueError(f"video_length should be equal to {a * bs + context_overlap} instead of {video_length},")
        for i in range(self.length + 1):
            num = bs if i != a else context_overlap

            latents = lip_pipeline.prepare_latents(
                batch_size,
                num_channels_latents,
                width,
                height,
                num,
                dtype,
                torch.device("cpu"),
                generator, )

            self.save(latents, i)

    def save(self, x, item):
        torch.save(x.cpu(), self.path_format.format(item))

    def __iter__(self):
        return next(self)

    def __next__(self):

        pre = self[0]
        for i in range(1, self.length + 1):
            now = self[i]
            res = torch.concatenate([pre, now[:, :, :self.context_overlap]], dim=2)
            pre = now
            yield res

    def __getitem__(self, item):

        return torch.load(self.path_format.format(item)).to(self.device)

    def __len__(self):
        return self.length
