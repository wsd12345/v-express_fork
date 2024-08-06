import argparse
import hashlib
import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import cv2 as cv
from PIL import Image

import utils.insert_path
from constant import BASE_PATH
from logger import init_log_conf
from models.pipeline import LipPipeline
from predict.audio_process import AudioFeature
from predict.face_process import FaceFeature, FaceIterOne
from predict.latents_process import Latents
from configs.lip_sync import CFG_LIP_SYNC as cfg
from predict.load_models import LipModels
from process.get_frames import VideoIter
from utils.get_mask import merge_face2image, get_mask_image

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")
init_log_conf("test_face_process", "debug")
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--base_cfg', type=str, help="",
                        default="../configs/lip_sync/cfg.yaml")

    args = parser.parse_args()
    logger.info(args)
    return args


def main(face_path, audio_dir, audio_sampling_rate):
    background_path = r""

    base_output_dir = BASE_PATH / cfg.PATH.OUTFILE

    dtype = torch.float16 if cfg.DTYPE == "fp16" else torch.float32
    device = torch.device("cuda:0") if "cuda" in cfg.DEVICE else torch.device("cpu")

    generator = torch.manual_seed(cfg.DIFFUSION.SEED)
    video_iter = VideoIter(face_path).__next__()
    lip_models = LipModels(cfg, cfg.DEVICE, cfg.DTYPE)

    md5_file = ""
    md5_name = hashlib.md5(face_path.encode()).hexdigest()
    md5_name = hashlib.md5((md5_file + md5_name).encode()).hexdigest()
    box = torch.load(BASE_PATH / cfg.PATH.CACHE / "face_process" / md5_name / "box.bin")
    background = np.array(Image.open(background_path).resize((1080, 1920)).convert("RGB")).astype(np.float32)[:, :,
                 ::-1]
    face_feature = FaceFeature(base_dir=BASE_PATH / cfg.PATH.CACHE / "face_process" / md5_name,
                               context_frames=cfg.DIFFUSION.CONTEXT_FRAMES,
                               context_overlap=cfg.DIFFUSION.CONTEXT_OVERLAP,
                               names=["kps_feature", "reference_control_writer_weight"]
                               )
    face_mask_iter = FaceIterOne(base_dir=BASE_PATH / cfg.PATH.CACHE / "face_process" / md5_name,
                                 names=["image_mask"], context_frames=1, context_overlap=1,
                                 ).__next__()

    lip_pipeline = LipPipeline(
        vae=lip_models.get("vae"),
        reference_net=lip_models.get("reference"),
        denoising_unet=lip_models.get("denoising"),
        v_kps_guider=lip_models.get("v_kps_guider"),
        audio_processor=None,
        audio_encoder=None,
        audio_projection=None,
        scheduler=lip_models.get("scheduler"),
    ).to(dtype=dtype, device=device)

    audio_feature = AudioFeature(cfg, sr=24000,
                                 base_dir=audio_dir,
                                 # names=os.listdir(audio_dir),
                                 names=['0.wav', '1.wav', '2.wav', '3.wav'],
                                 audio_processor=lip_models.get('audio_processor'),
                                 audio_encoder=lip_models.get('audio_encoder'),
                                 audio_projection=lip_models.get('audio_projection'),
                                 )
    index_face = 0
    for i, (audio_embeddings, video_length, init_video_length) in enumerate([audio_feature.get_all_audio_embeddings()]):
        latents = Latents(cfg,
                          BASE_PATH / cfg.PATH.CACHE / "latents" / md5_name,
                          lip_pipeline,
                          batch_size=1,
                          num_channels_latents=lip_pipeline.denoising_unet.in_channels,
                          width=cfg.WIDTH_NET,
                          height=cfg.HEIGHT_NET,
                          video_length=video_length,
                          dtype=dtype,
                          device=device,
                          generator=generator)

        num_face = math.ceil((video_length - face_feature.pre_length) / cfg.MAX_NUM_CACHE)
        face_feature.set_indexes(list(range(index_face, index_face + num_face)), video_length - init_video_length)

        lip_pipeline(
            audio_embeddings=audio_embeddings,
            face_feature_obj=face_feature,
            latents_obj=latents,
            video_length=video_length,
            num_inference_steps=cfg.DIFFUSION.NUM_INFERENCE_STEPS,
            guidance_scale=cfg.DIFFUSION.GUIDANCE_SCALE,
            context_frames=cfg.DIFFUSION.CONTEXT_FRAMES,
            context_overlap=cfg.DIFFUSION.CONTEXT_OVERLAP,
            reference_attention_weight=cfg.DIFFUSION.REFERENCE_ATTENTION_WEIGHT,
            audio_attention_weight=cfg.DIFFUSION.AUDIO_ATTENTION_WEIGHT,
            generator=generator,
        )
        (BASE_PATH / cfg.PATH.TEMP / md5_name).mkdir(parents=True, exist_ok=True)
        writer = cv.VideoWriter(str(BASE_PATH / cfg.PATH.TEMP / md5_name / f"{i}.mp4"),
                                cv.VideoWriter_fourcc(*'mp4v'), 25, (1080, 1920))

        for i_face_gen, face_gen in enumerate(lip_pipeline.decode_latents(latents)):
            full_image = next(video_iter)
            mask = next(face_mask_iter)
            fin = merge_face2image(full_image, face_gen[:, :, ::-1] * 255, box, 50)  # todo 50
            # res = get_mask_image(fin, mask, background)
            res = fin.astype(np.uint8)
            writer.write(res)
            init_video_length -= 1
            if init_video_length == 0:
                break
        writer.release()
        video_path = BASE_PATH / cfg.PATH.TEMP / md5_name / f'{i}.mp4'
        logger.info(f"save temp video in {video_path}")
        index_face += num_face


if __name__ == '__main__':
    face_path = r" "
    audio_dir = r" "
    main(face_path, audio_dir, 0)
