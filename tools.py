import cv2
import pytz
import datetime
import numpy as np

from config import REID_IMAGE_W, REID_IMAGE_H


TIME_ZONE = 'Europe/Moscow'


def get_date_now_formatted() -> str:
    """ Get now datetime in %Y/%m/%d_%H:%M:%S format. """
    tz = pytz.timezone(TIME_ZONE)
    now = datetime.datetime.now(tz)
    return now.strftime("%Y/%m/%d_%H:%M:%S")


def reid_img_preproc(src_image: np.ndarray) -> np.ndarray:
    """ Preproc image to Reid. """
    image = cv2.resize(src_image, (REID_IMAGE_W, REID_IMAGE_H))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.reshape(image, (1, REID_IMAGE_H, REID_IMAGE_W, 3)).astype(float)
    return image
