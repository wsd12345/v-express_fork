import argparse
import hashlib
import logging
import warnings

import utils.insert_path
from configs import update_config
from configs.lip_sync import CFG_LIP_SYNC as cfg
from constant import BASE_PATH, DEVICE
from logger import init_log_conf
from predict.face_process import save_data
from predict.load_models import LipModels

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


def main(faces_path):
    args = get_args()
    update_config(cfg, args.base_cfg)

    # md5_file = get_file_md5([args.face_cfg])
    md5_file = ""

    lip_models = LipModels(cfg, cfg.DEVICE, cfg.DTYPE)

    base_path = BASE_PATH / cfg.PATH.CACHE / "face_process"
    for face_path in faces_path:
        md5_name = hashlib.md5(face_path.encode()).hexdigest()
        md5_name = hashlib.md5((md5_file + md5_name).encode()).hexdigest()
        save_path = base_path / md5_name
        save_data(
            cfg,
            path_face=face_path,
            save_path=save_path,
            croper=lip_models.get("croper"),
            vae=lip_models.get("vae"),
            face_analysis_net=lip_models.get("face_analysis"),
            reference_net=lip_models.get("reference"),
            v_kps_guider_net=lip_models.get("v_kps_guider")
        )

        logger.info("end face_process")


if __name__ == '__main__':
    main([
        r" "
    ])
