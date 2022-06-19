import cv2
import pytz
import datetime
import numpy as np

from config import BLACK_MARGINS
from config import REID_IMAGE_W, REID_IMAGE_H
from config import REID_IMAGE_SHAPE, REID_IMAGE_DTYPE


TIME_ZONE = 'Europe/Moscow'


def get_date_now_formatted() -> str:
    """ Get now datetime in %Y-%m-%d_%H:%M:%S:%f format. """
    tz = pytz.timezone(TIME_ZONE)
    now = datetime.datetime.now(tz)
    return now.strftime("%Y-%m-%d_%H:%M:%S:%f")


def reid_img_preproc(src_image: np.ndarray) -> np.ndarray:
    """ Preproc image to Reid. """
    image = add_black_margins(src_image) if BLACK_MARGINS else src_image
    image = cv2.resize(image, (REID_IMAGE_W, REID_IMAGE_H))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.reshape(image, (1, REID_IMAGE_H, REID_IMAGE_W, 3)).astype(float)
    return image


def add_black_margins(src_image: np.ndarray) -> np.ndarray:
    """ Image enlargement with the addition of black margins. """
    height, width, depth = src_image.shape
    required_height = round(REID_IMAGE_H * width / REID_IMAGE_W)
    if required_height > height:
        half_diff = round((required_height - height) / 2)
        new_image = np.zeros((required_height, width, depth), src_image.dtype)
        new_image[half_diff:half_diff+height, 0:width] = src_image
    else:
        required_width = round(REID_IMAGE_W * height / REID_IMAGE_H)
        half_diff = round((required_width - width) / 2)
        new_image = np.zeros((height, required_width, depth), src_image.dtype)
        new_image[0:height, half_diff:half_diff+width] = src_image
    return new_image


def reid_img_revert(src_image: np.ndarray) -> np.ndarray:
    """ Revert image from reid format. """
    image = np.reshape(src_image, (REID_IMAGE_H, REID_IMAGE_W, 3)).astype(np.dtype("uint8"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def bytes_to_image(img_bytes: bytes) -> np.ndarray:
    """ Serialized bytes to numpy array (image). """
    return np.ndarray(shape=REID_IMAGE_SHAPE, dtype=REID_IMAGE_DTYPE, buffer=img_bytes)


def get_iou(bbox_1, bbox_2) -> float:
    """ Get iou of two bboxes. """
    max_x_min = max(bbox_1[1], bbox_2[1])
    max_y_min = max(bbox_1[0], bbox_2[0])
    min_x_max = min(bbox_1[3], bbox_2[3])
    min_y_max = min(bbox_1[2], bbox_2[2])

    inter_area = max(0, min_x_max - max_x_min) * max(0, min_y_max - max_y_min)

    bbox_1_area = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])
    bbox_2_area = (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])

    iou = float(inter_area) / float(bbox_1_area + bbox_2_area - inter_area) * 100
    return iou


def get_xy_max_diff(bbox_1, bbox_2) -> float:
    max_x = max(bbox_1[2] - bbox_1[0], bbox_2[2] - bbox_2[0])
    max_y = max(bbox_1[3] - bbox_1[1], bbox_2[3] - bbox_2[1])
    min_x = min(bbox_1[2] - bbox_1[0], bbox_2[2] - bbox_2[0])
    min_y = min(bbox_1[3] - bbox_1[1], bbox_2[3] - bbox_2[1])
    # return max(abs(max_x - min_x) / max_x, abs(max_y - min_y) / max_y) * 100
    return abs(max_y - min_y) / max_y * 100
