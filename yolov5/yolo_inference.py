import cv2
import torch
import numpy as np

from typing import List
from pathlib import Path


class Yolo:
    def __init__(self):
        device = "0" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", device=device)
        model.classes = [0]  # only humans
        self.model = model

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Infer people on image.

        :param image: image in numpy format.
        :return: numpy array with bboxes ([x_min, y_min, x_max, y_max])
        """
        results = self.model(image)
        results_pd = results.pandas().xyxy[0]
        bboxes = np.array(results_pd[['xmin', 'ymin', 'xmax', 'ymax']]).astype(int)
        return bboxes

    def create_person_images(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Create cropped images with detected people on source image.
        :param image: image in numpy format.
        :return: list with cropped images on numpy format.
        """
        bboxes = self.infer(image)
        images = [self._crop_image(image, bbox) for bbox in bboxes]
        return images

    @staticmethod
    def _crop_image(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Crop bbox from image.
        :param image: image in numpy format.
        :param bbox: numpy array ([x_min, y_min, x_max, y_max])
        :return: cropped image in numpy format.
        """
        x_min, y_min, x_max, y_max = bbox
        crop_im = image[y_min:y_max, x_min:x_max]
        return crop_im


if __name__ == "__main__":
    image = cv2.imread(str(Path.joinpath(Path(__file__).parent.parent, "test_data/test_we.png")))

    yolo = Yolo()
    # Just infer.
    print("bboxes:\n", yolo.infer(image))
    # Infer and create cropped images with people.
    person_images = yolo.create_person_images(image)
    for img in person_images:
        cv2.imshow("test", img)
        cv2.waitKey(0)
