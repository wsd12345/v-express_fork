import logging

import torch

from omegaconf import OmegaConf
from insightface.app import FaceAnalysis
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import is_xformers_available
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from yacs.config import CfgNode
from typing import Optional, Callable

from models.croper import Croper
from third_part.V_Express.modules import (
    UNet2DConditionModel,
    UNet3DConditionModel,
    VKpsGuider,
    AudioProjection
)

logger = logging.getLogger(__name__)


def load_reference_net(unet_config_path, reference_net_path):
    reference_net = UNet2DConditionModel.from_config(unet_config_path)
    reference_net.load_state_dict(torch.load(reference_net_path, map_location="cpu"), strict=False)
    logger.info(f'Loaded weights of Reference Net from {reference_net_path}.')
    return reference_net


def load_denoising_unet(inf_config_path, unet_config_path, denoising_unet_path, motion_module_path):
    inference_config = OmegaConf.load(inf_config_path)
    denoising_unet = UNet3DConditionModel.from_config_2d(
        unet_config_path,
        unet_additional_kwargs=inference_config.unet_additional_kwargs,
    )
    denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)
    logger.info(f'Loaded weights of Denoising U-Net from {denoising_unet_path}.')

    denoising_unet.load_state_dict(torch.load(motion_module_path, map_location="cpu"), strict=False)
    logger.info(f'Loaded weights of Denoising U-Net Motion Module from {motion_module_path}.')

    return denoising_unet


def load_v_kps_guider(v_kps_guider_path):
    v_kps_guider = VKpsGuider(320, block_out_channels=(16, 32, 96, 256))
    v_kps_guider.load_state_dict(torch.load(v_kps_guider_path, map_location="cpu"))
    logger.info(f'Loaded weights of V-Kps Guider from {v_kps_guider_path}.')
    return v_kps_guider


def load_audio_projection(
        audio_projection_path,
        inp_dim: int,
        mid_dim: int,
        out_dim: int,
        inp_seq_len: int,
        out_seq_len: int,
):
    audio_projection = AudioProjection(
        dim=mid_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=out_seq_len,
        embedding_dim=inp_dim,
        output_dim=out_dim,
        ff_mult=4,
        max_seq_len=inp_seq_len,
    )
    audio_projection.load_state_dict(torch.load(audio_projection_path, map_location='cpu'))
    logger.info(f'Loaded weights of Audio Projection from {audio_projection_path}.')
    return audio_projection


def get_scheduler(inference_config_path):
    inference_config = OmegaConf.load(inference_config_path)
    scheduler_kwargs = OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**scheduler_kwargs)
    return scheduler


class Models(dict):
    model_names = ()

    def __init__(self, cfg: CfgNode, device: str, dtype: str = "fp32"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = torch.float16 if dtype == 'fp16' else torch.float32

    def set(self, model):
        return model.to(dtype=self.dtype, device=self.device)

    def get(self, key) -> Optional[Callable]:
        if key not in self.model_names:
            return None
        value = getattr(self, key, None)
        if value is None:
            value = getattr(self, f"_{self.__class__.__name__}__{key}")()
            setattr(self, key, value)
        return value


class LipModels(Models):
    model_names = (
        'audio_encoder',
        'audio_processor',
        'audio_projection',
        'croper',
        'denoising',
        'face_analysis',
        'reference',
        'scheduler',
        'v_kps_guider',
        'vae')

    def __croper(self):
        return Croper(self.cfg.MODEL_PATH.CROPER) if self.cfg.MODEL_PATH.CROPER else None

    def __scheduler(self):
        # inference_config_path = './inference_v2.yaml'
        return get_scheduler(self.cfg.CONFIG_PATH.INFERENCE)

    def __vae(self):
        return self.set(AutoencoderKL.from_pretrained(self.cfg.MODEL_PATH.VAE))

    def __reference(self):
        model = self.set(load_reference_net(self.cfg.CONFIG_PATH.UNET,
                                            self.cfg.MODEL_PATH.REFERENCE
                                            ))
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        return model

    def __denoising(self):
        model = self.set(load_denoising_unet(
            self.cfg.CONFIG_PATH.INFERENCE, self.cfg.CONFIG_PATH.UNET,
            self.cfg.MODEL_PATH.DENOISING, self.cfg.MODEL_PATH.MOTION)
        )
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        return model

    def __v_kps_guider(self):
        return self.set(load_v_kps_guider(self.cfg.MODEL_PATH.V_KPS_GUIDER))

    def __audio_processor(self):
        return Wav2Vec2Processor.from_pretrained(self.cfg.MODEL_PATH.AUDIO_ENCODER)

    def __audio_encoder(self):
        return self.set(Wav2Vec2Model.from_pretrained(self.cfg.MODEL_PATH.AUDIO_ENCODER))

    def __audio_projection(self):
        denoising_unet = self.get("denoising")
        return self.set(load_audio_projection(
            self.cfg.MODEL_PATH.AUDIO_PROJECTION,
            inp_dim=denoising_unet.config.cross_attention_dim,
            mid_dim=denoising_unet.config.cross_attention_dim,
            out_dim=denoising_unet.config.cross_attention_dim,
            inp_seq_len=2 * (2 * self.cfg.AUDIO.NUM_PAD + 1),
            out_seq_len=2 * self.cfg.AUDIO.NUM_PAD + 1)
        )

    def __face_analysis(self):
        app = FaceAnalysis(
            providers=['CUDAExecutionProvider' if 'cuda' in self.device else 'CPUExecutionProvider'],
            # provider_options=[{'device_id': args.gpu_id}] if args.device == 'cuda' else [],
            root=self.cfg.MODEL_PATH.INSIGHTFACE,
        )
        app.prepare(ctx_id=0, det_size=(self.cfg.HEIGHT_NET, self.cfg.WIDTH_NET))
        return app
