import cv2
import random
import numpy as np

from typing import Tuple
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


FONT = ImageFont.truetype('fonts/Fallout2Cyr.ttf', 32)
COLORS = {-1: (255, 255, 255),
          0: (0, 0, 255),
          1: (0, 255, 0),
          2: (0, 128, 128),
          3: (128, 0, 0),
          4: (128, 0, 128),
          5: (128, 128, 0),
          6: (128, 128, 128),
          7: (0, 0, 64)}


def mark_frame(image: np.ndarray, objects: list) -> np.ndarray:
    """
    Draw objects on image.

    :param image: source image.
    :param objects: list of objects.
    :return: image with objects.
    """
    h, w, d = image.shape
    screen_h = 1080
    screen_w = 1920
    # Thickness of lines.
    scale = np.min([float(screen_h) / float(h), float(screen_w) / float(w)])
    for obj in objects:
        image = draw_object_rect(obj, image, scale)
    # Draw objects labels.
    for obj in objects:
        image = draw_object(image, obj)
    return image


def draw_object_rect(obj: dict, src_img: np.ndarray, scale: np.float) -> np.ndarray:
    """
    Draw labeled object on image.

    :param obj: dictionary with data about object.
    :param src_img: image to draw edging.
    :param added_img: image to draw fills.
    :param scale: scale of thickness edging.
    :return: marked image.
    """
    color = get_color(obj["p_id"])
    coords = convert_coords(obj["bbox"])
    # Draw rectangle.
    cv2.polylines(src_img, [coords], True, color, thickness=int(4.0 / scale))
    return src_img


def get_color(id: int) -> Tuple[int, int, int]:
    """
    Return the color of object.

    :param id: id of object.
    :return: color.
    """
    if id in COLORS.keys():
        return COLORS[id]
    new_color = COLORS[0]
    colors_values = COLORS.values()
    while new_color in colors_values:
        new_color = (random.randint(20, 230), random.randint(20, 230), random.randint(20, 230))
    COLORS[len(COLORS)] = new_color
    return new_color


def convert_coords(coords: list) -> np.ndarray:
    """
    Convert coords from (x_min, y_min, x_max, y_max) to ([x1, y1], [x2, y2], [x3, y3], [x4, y4]).

    :param coords: coords of points.
    :return: converted coords.
    """
    x_min, y_min, x_max, y_max = coords
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
    xy = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype(int)  # Polygon
    return xy


def draw_object(image: np.ndarray, obj: dict) -> np.ndarray:
    person_id = obj["p_id"]
    text_to_draw = f"Человек {person_id}"

    # Draw text background.
    font_w, font_h = FONT.getsize(text_to_draw)
    x_min, y_min, x_max, y_max = obj["bbox"]
    color = get_color(person_id)
    cv2.rectangle(image, (x_min, y_min - font_h - 2), (x_min + font_w, y_min), color, -1)
    # Draw text.
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x_min, y_min - font_h - 2), text_to_draw, font=FONT, fill=(255, 255, 255, 1))
    image = np.array(img_pil)
    return image
