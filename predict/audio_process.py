import logging
import math
from pathlib import Path
from typing import List

import torch
import torchaudio
import torchvision

from models.pipeline import AudioPipeline
from process.get_audio import get_audio_waveform

logger = logging.getLogger(__name__)


class AudioFeature:
    def __init__(self,
                 cfg,
                 sr: int,
                 base_dir: str,
                 names: List[str],
                 audio_processor,
                 audio_encoder,
                 audio_projection,
                 silence=0
                 ):
        self.silence = silence
        self.cfg = cfg
        self.sr = sr
        self.num_pad_audio_frames = self.cfg.AUDIO.NUM_PAD
        self.do_classifier_free_guidance = self.cfg.DIFFUSION.GUIDANCE_SCALE > 1
        self.base_dir = Path(base_dir)
        self.names = names
        self.n = len(self.names)
        self.pipeline = AudioPipeline(audio_processor,
                                      audio_encoder,
                                      audio_projection)

    def __iter__(self):
        return next(self)

    def __next__(self):
        for i in range(len(self.names)):
            yield self[i]

    def __getitem__(self, item):
        if item > len(self.names):
            raise IndexError("list index out of range")

        # audio = torch.load(self.base_dir / self.names[item])
        _, audio, meta_info = torchvision.io.read_video(str(self.base_dir / self.names[item]), pts_unit='sec')
        audio = audio.type(torch.float32)
        self.sr = meta_info['audio_fps']
        audio_waveform = self.resample(audio.type(torch.float32), meta_info['audio_fps'])
        audio_waveform, video_length, init_video_length = self.calc_audio_length(
            audio_waveform, self.cfg.AUDIO.SR, self.cfg.FPS,
            self.cfg.DIFFUSION.CONTEXT_FRAMES, self.cfg.DIFFUSION.CONTEXT_OVERLAP,
            silence=self.silence,
        )
        audio_embeddings = self.pipeline.prepare_audio_embeddings(
            audio_waveform, video_length,
            self.num_pad_audio_frames, self.do_classifier_free_guidance)
        torch.cuda.empty_cache()
        return audio_embeddings, video_length, init_video_length

    def __len__(self):
        return self.n

    def get_all_audio_embeddings(self, silence=0):
        audio_waveform_list = []
        for i in self.names:
            _, audio, meta_info = torchvision.io.read_video(str(self.base_dir / i), pts_unit='sec')
            audio_waveform = self.resample(audio.type(torch.float32), meta_info['audio_fps'])
            audio_waveform_list.append(audio_waveform)

        audio_waveform = torch.concatenate(audio_waveform_list, dim=1)

        audio_waveform, video_length, init_video_length = self.calc_audio_length(
            audio_waveform, self.cfg.AUDIO.SR, self.cfg.FPS,
            self.cfg.DIFFUSION.CONTEXT_FRAMES, self.cfg.DIFFUSION.CONTEXT_OVERLAP,
            silence
        )

        audio_embeddings = self.pipeline.prepare_audio_embeddings(
            audio_waveform, video_length,
            self.num_pad_audio_frames, self.do_classifier_free_guidance)
        torch.cuda.empty_cache()
        return audio_embeddings, video_length, init_video_length

    def resample(self, audio_waveform, sr):
        if sr != self.cfg.AUDIO.SR:
            audio_waveform = torchaudio.functional.resample(
                audio_waveform,
                orig_freq=sr,
                new_freq=self.cfg.AUDIO.SR,
            )

        return audio_waveform

    @staticmethod
    def calc_audio_length(audio_waveform, sr, fps,
                          contest_frame=12, context_overlap=4,
                          silence=0):
        audio_waveform = audio_waveform.mean(dim=0)

        duration = audio_waveform.shape[0] / sr + silence

        init_video_length = math.ceil(duration * fps)

        t = init_video_length % (contest_frame - context_overlap)
        num_pad = context_overlap if t == 0 else (contest_frame - t)

        audio_pad = torch.zeros((int(sr * (silence + num_pad / fps)),),
                                dtype=audio_waveform.dtype, device=audio_waveform.device)
        audio_waveform = torch.concatenate([audio_waveform, audio_pad])
        num_pad += init_video_length
        logger.info(f'The corresponding video length is {num_pad}, origin {init_video_length}')

        return audio_waveform, num_pad, init_video_length
