import logging
import math
import torch
import torchaudio
import torchvision
from yacs.config import CfgNode
from typing import Tuple

logger = logging.getLogger(__name__)


def get_audio_waveform(cfg: CfgNode,
                       audio_waveform,
                       audio_sampling_rate,
                       silence: int = 0,
                       ) -> Tuple[torch.Tensor, int, int]:
    # _, audio_waveform, meta_info = torchvision.io.read_video(cfg.AUDIO.PATH, pts_unit='sec')
    #
    # audio_waveform = audio_waveform.type(torch.float32)
    # audio_sampling_rate = meta_info['audio_fps']

    # logger.info(f'Length of audio is {audio_waveform.shape[1]} with the sampling rate of {audio_sampling_rate}.')
    if audio_sampling_rate != cfg.AUDIO.SR:
        audio_waveform = torchaudio.functional.resample(
            audio_waveform,
            orig_freq=audio_sampling_rate,
            new_freq=cfg.AUDIO.SR,
        )

    audio_waveform = audio_waveform.mean(dim=0)

    duration = audio_waveform.shape[0] / cfg.AUDIO.SR + silence

    init_video_length = math.ceil(duration * cfg.FPS)

    fps = cfg.FPS
    t = init_video_length % (cfg.DIFFUSION.CONTEXT_FRAMES - cfg.DIFFUSION.CONTEXT_OVERLAP)
    num_pad = cfg.DIFFUSION.CONTEXT_OVERLAP if t == 0 else (cfg.DIFFUSION.CONTEXT_FRAMES - t)

    audio_pad = torch.zeros((int(cfg.AUDIO.SR * (silence + num_pad / fps)),),
                            dtype=audio_waveform.dtype, device=audio_waveform.device)
    audio_waveform = torch.concatenate([audio_waveform, audio_pad])
    num_pad += init_video_length
    logger.info(f'The corresponding video length is {num_pad}, origin {init_video_length}')

    return audio_waveform, num_pad, init_video_length
