from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

from constant import BASE_PATH, DEVICE

_C = CN()

# ---------- MODEL_PATH ---------- #
_C.MODEL_PATH = CN()

_C.MODEL_PATH.AUDIO_ENCODER = str(BASE_PATH / 'model_ckpts/wav2vec2-base-960h/')
_C.MODEL_PATH.AUDIO_PROJECTION = str(BASE_PATH / "model_ckpts/v-express/audio_projection.bin")
_C.MODEL_PATH.CROPER = str(BASE_PATH / "model_ckpts/shape_predictor_68_face_landmarks.dat")
_C.MODEL_PATH.DENOISING = str(BASE_PATH / "model_ckpts/v-express/denoising_unet.bin")
_C.MODEL_PATH.INSIGHTFACE = str(BASE_PATH / "model_ckpts/insightface_models/")
_C.MODEL_PATH.MOTION = str(BASE_PATH / "model_ckpts/v-express/motion_module.bin")
_C.MODEL_PATH.REFERENCE = str(BASE_PATH / "model_ckpts/v-express/reference_net.bin")
_C.MODEL_PATH.V_KPS_GUIDER = str(BASE_PATH / "model_ckpts/v-express/v_kps_guider.bin")
_C.MODEL_PATH.VAE = str(BASE_PATH / "model_ckpts/sd-vae-ft-mse/")

# ---------- CONFIG_PATH ---------- #
_C.CONFIG_PATH = CN()
_C.CONFIG_PATH.UNET = str(BASE_PATH / "model_ckpts/stable-diffusion-v1-5/unet/config.json")
_C.CONFIG_PATH.INFERENCE = str(BASE_PATH / "model_ckpts/inference_v2.yaml")

# ---------- PATH ---------- #
_C.PATH = CN()
_C.PATH.TEMP = str(BASE_PATH / "temp")
_C.PATH.CACHE = str(BASE_PATH / "cache")
_C.PATH.VIDEO = ""
_C.PATH.AUDIO = ""
_C.PATH.OUTFILE = ""
_C.PATH.BACKGROUND = ""

# ---------- OUTPUT ---------- #

# ---------- AUDIO ---------- #
_C.AUDIO = CN()
_C.AUDIO.SR = 16000
_C.AUDIO.NUM_PAD = 2

# ---------- DIFFUSION ---------- #
_C.DIFFUSION = CN()

_C.DIFFUSION.SEED = 42
_C.DIFFUSION.CONTEXT_FRAMES = 12
_C.DIFFUSION.CONTEXT_OVERLAP = 4
_C.DIFFUSION.GUIDANCE_SCALE = 2.5
_C.DIFFUSION.NUM_INFERENCE_STEPS = 15
_C.DIFFUSION.REFERENCE_ATTENTION_WEIGHT = 0.95
_C.DIFFUSION.AUDIO_ATTENTION_WEIGHT = 3

# ----- INPUT -----
_C.FPS = 25
_C.CROP = [0, -1, 0, -1]
_C.VERBOSE = False
_C.DTYPE = "fp16"

# CROPER
_C.WIDTH_FACE = -1
_C.HEIGHT_FACE = -1
# NETWORK
_C.WIDTH_NET = 512
_C.HEIGHT_NET = 512
# OUTPUT
_C.WIDTH_OUT = 1080
_C.HEIGHT_OUT = 1920

_C.MAX_NUM_CACHE = _C.DIFFUSION.CONTEXT_FRAMES - _C.DIFFUSION.CONTEXT_OVERLAP
_C.DEVICE = DEVICE
