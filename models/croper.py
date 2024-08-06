"""

copy and revise from  video_retalking
https://github.com/OpenTalker/video-retalking.git
"""
import numpy as np
from PIL import Image
import dlib


class Croper:
    def __init__(self, path_of_lm):
        # download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor(path_of_lm)

    def get_landmark(self, img_np):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        detector = dlib.get_frontal_face_detector()
        dets = detector(img_np, 1)
        if len(dets) == 0:
            return None
        d = dets[0]
        # Get the landmarks/parts for the face in box d.
        shape = self.predictor(img_np, d)
        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        return lm

    def align_face(self, img, lm, output_size=1024):
        """
        :param filepath: str
        :return: PIL Image
        """
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            quad -= crop[0:2]

        # Transform.
        quad = (quad + 0.5).flatten()
        lx = max(min(quad[0], quad[2]), 0)
        ly = max(min(quad[1], quad[7]), 0)
        rx = min(max(quad[4], quad[6]), img.size[0])
        ry = min(max(quad[3], quad[5]), img.size[0])

        # Save aligned image.
        return crop, [lx, ly, rx, ry]

    def crop(self, img_np_list, xsize=512, out_size=-1):  # first frame for all video

        idx = 0
        while idx < len(img_np_list) // 2 + 1:
            img_np = img_np_list[idx]
            lm = self.get_landmark(img_np)
            if lm is not None:
                break  # can detect face
            idx += 1
        if lm is None:
            return None

        crop, quad = self.align_face(img=Image.fromarray(img_np), lm=lm, output_size=xsize)
        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = [cly + ly, cly + ry, clx + lx, clx + rx]

        if out_size == -1:
            ny1, ny2, nx1, nx2 = oy1, oy2, ox1, ox2
        else:
            x = (ox1 + ox2) // 2
            y = (oy1 + oy2) // 2
            l = out_size // 2
            ny1, ny2, nx1, nx2 = y - l, y + l, x - l, x + l
            if ox2 - ox1 > out_size or oy2 - oy1 > out_size:
                raise Exception(f"face_size = {(ox2 - ox1, oy2 - oy1)}, more than {(out_size, out_size)}")

        img_np_list = self.update(img_np_list, [ny1, ny2, nx1, nx2])
        return img_np_list, crop, quad, [ny1, ny2, nx1, nx2]

    @staticmethod
    def update(img_np_list, box):
        ny1, ny2, nx1, nx2 = box
        for _i in range(len(img_np_list)):
            img_np_list[_i] = img_np_list[_i][ny1:ny2, nx1:nx2]
        return img_np_list
