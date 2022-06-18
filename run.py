import cv2
import numpy as np

import db

from reid import Reid
from yolov5 import Yolo
from tools import get_iou
from tools import get_xy_max_diff
from tools import reid_img_preproc
from tools import get_date_now_formatted
from config import XY_THRESHOLD, IOU_THRESHOLD, MAX_IOU_THRESHOLD
from config import SOURCE_VIDEO_FILE_PATH, SAVE_RECORDS, REC_PATH, EACH_FRAME


class MainClass:
    def __init__(self) -> None:
        """ Init nets. """
        # self.reid = Reid()
        self.yolo = Yolo()
        self.prev_bboxes_data = []  # condition storage
        self.temp_prev_bboxes_data = []  # temporary condition storage

    def file_capturing(self) -> None:
        cap = cv2.VideoCapture(SOURCE_VIDEO_FILE_PATH)
        i_frame = 0
        while True:
            # Read frame from video.
            _, frame = cap.read()
            # Skip frames.
            i_frame += 1
            if i_frame % EACH_FRAME != 0:
                continue

            # Work with frame.
            result_frame = self.frame_proc(frame)
            cv2.imshow("Q for exit", result_frame)
            # Save `result_frame` to `REC_PATH`
            if SAVE_RECORDS:
                self.save_frame(result_frame)

            key = cv2.waitKey(1)
            if key & 0xFF in [ord('q'), ord('Q'), ord('й'), ord('Й')]:
                break
        # Close capturing.
        cap.release()

    def frame_proc(self, frame: np.ndarray) -> np.ndarray:
        person_images, bboxes = self.yolo.create_person_images(frame)
        for person_image, bbox in zip(person_images, bboxes):
            person_id = self.person_proc(person_image, bbox)
        self.prev_bboxes_data = self.temp_prev_bboxes_data
        return frame

    def person_proc(self, person_image: np.ndarray, bbox: np.ndarray) -> int:
        # Create list of tuples (iou, bbox).
        iou = [(get_iou(prev_bbox_data["bbox"], bbox), prev_bbox_data) for prev_bbox_data in self.prev_bboxes_data]
        if len(iou) == 0:  # if new person (can be only part of body)
            self.temp_prev_bboxes_data.append({"p_id": -1, "bbox": bbox})
            return -1
        max_iou = sorted(iou, key=lambda x: x[0])[-1]
        if max_iou[0] < IOU_THRESHOLD:  # if small iou - new person (can be only part of body)
            self.temp_prev_bboxes_data.append({"p_id": -1, "bbox": bbox})
            return -1
        xy_20 = get_xy_max_diff(bbox, max_iou[1]["bbox"]) < XY_THRESHOLD
        # Sophisticated logic.
        if xy_20 and max_iou[1]["p_id"] == -1:  # second arrival of a new person (small bbox change)
            return self.reid_comparator(person_image)
        elif xy_20 and max_iou[1]["p_id"] != -1:  # another arrival of an already known person (small bbox change)
            return max_iou[1]["p_id"]
        elif not xy_20 and max_iou[1]["p_id"] == -1:  # big bbox change and unknown person
            self.temp_prev_bboxes_data.append({"p_id": -1, "bbox": bbox})
            return -1
        elif not xy_20 and max_iou[1]["p_id"] != -1:  # perhaps known, but because of big bbox change
            if max_iou[0] > MAX_IOU_THRESHOLD:
                return max_iou[1]["p_id"]  # if the intersection is big - believe that the known person
            else:
                return self.reid_comparator(person_image)  # else: new or not, reid will check

    def reid_comparator(self, person_image: np.ndarray) -> int:
        reid_image = reid_img_preproc(person_image)

        return -1


    @staticmethod
    def save_frame(frame: np.ndarray) -> None:
        """ Save frame to recordings directory. """
        file_name = f"{get_date_now_formatted()}.jpg"
        cv2.imwrite(f"{REC_PATH}/{file_name}", frame)


if __name__ == "__main__":
    main = MainClass()
    main.file_capturing()
