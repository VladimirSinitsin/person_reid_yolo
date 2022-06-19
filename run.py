import cv2
import shutil
import numpy as np

import db

from reid import Reid
from yolov5 import Yolo
from tools import get_iou
from tools import mark_frame
from tools import get_y_max_diff
from tools import reid_img_preproc
from tools import get_date_now_formatted
from config import SOURCE_VIDEO_FILE_PATH, SAVE_RECORDS, REC_PATH, EACH_FRAME
from config import XY_THRESHOLD, IOU_THRESHOLD, MAX_IOU_THRESHOLD, PERSON_REID_THRESHOLD


class MainClass:
    def __init__(self) -> None:
        """ Init nets. """
        self.yolo = Yolo()
        self.reid = Reid()
        self.prev_bboxes_data = []  # condition storage
        self.temp_prev_bboxes_data = []  # temporary condition storage

        if SAVE_RECORDS:
            # Create directory.
            if REC_PATH.exists():
                shutil.rmtree(REC_PATH)
            REC_PATH.mkdir(exist_ok=True)

    def file_capturing(self) -> None:
        cap = cv2.VideoCapture(SOURCE_VIDEO_FILE_PATH)
        i_frame = 0
        while True:
            # Read frame from video.
            ret, frame = cap.read()
            if frame is None:
                break
            if not ret:
                continue
            # Skip frames.
            i_frame += 1
            if i_frame % EACH_FRAME != 0:
                continue

            # Work with frame.
            result_frame = self.frame_proc(frame)
            result_frame = cv2.resize(result_frame, (1280, 720))
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
        self.temp_prev_bboxes_data = []
        person_objects = []
        person_images, bboxes = self.yolo.create_person_images(frame)
        for person_image, bbox in zip(person_images, bboxes):
            person_image = reid_img_preproc(person_image)
            person_id = self.person_proc(person_image, bbox)
            self.temp_prev_bboxes_data.append({"p_id": person_id, "bbox": bbox})
            if person_id != -1:
                db.insert_image_data(person_id=person_id, image=person_image)
            person_objects.append({"p_id": person_id, "bbox": bbox})
        self.prev_bboxes_data = self.temp_prev_bboxes_data
        result_frame = mark_frame(frame, person_objects)
        return result_frame

    def person_proc(self, person_image: np.ndarray, bbox: np.ndarray) -> int:
        # Create list of tuples (iou, bbox).
        iou = [(get_iou(prev_bbox_data["bbox"], bbox), prev_bbox_data) for prev_bbox_data in self.prev_bboxes_data]
        if len(iou) == 0:  # if new person (can be only part of body)
            return -1
        max_iou = sorted(iou, key=lambda x: x[0])[-1]
        if max_iou[0] < IOU_THRESHOLD:  # if small iou - new person (can be only part of body)
            return -1
        y_thr = get_y_max_diff(bbox, max_iou[1]["bbox"]) < XY_THRESHOLD
        # Sophisticated logic.
        if y_thr and max_iou[1]["p_id"] == -1:  # second arrival of a new person (small bbox change)
            return self.reid_comparator(person_image)
        elif y_thr and max_iou[1]["p_id"] != -1:  # another arrival of an already known person (small bbox change)
            return max_iou[1]["p_id"]
        elif not y_thr and max_iou[1]["p_id"] == -1:  # big bbox change and unknown person
            return -1
        elif not y_thr and max_iou[1]["p_id"] != -1:  # perhaps known, but because of big bbox change
            if max_iou[0] > MAX_IOU_THRESHOLD:
                return max_iou[1]["p_id"]  # if the intersection is big - believe that the known person
            else:
                return self.reid_comparator(person_image)  # else: new or not, reid will check

    def reid_comparator(self, person_image: np.ndarray) -> int:
        """ Compare `person_image` with other people in DB using Reid. """
        max_person_id = db.select_max_person_id()
        if max_person_id == -1:  # if db is empty
            return 0
        coincidences = []
        for person_id in range(max_person_id + 1):
            person_images_data = db.select_curr_images_data(person_id=person_id)
            coincidence = self.persons_compare(person_image, person_images_data)
            coincidences.append(coincidence)
        coincidences = np.array(coincidences)
        max_coincidence, ind = np.max(coincidences), np.argmax(coincidences)
        # Set id of person from DB or new.
        curr_person_id = int(ind) if max_coincidence >= PERSON_REID_THRESHOLD else max_person_id + 1
        return curr_person_id

    def persons_compare(self, person_image: np.ndarray, other_images_data: list) -> float:
        """ Compare `person_image` with list of images with other people using Reid. """
        same = 0
        count = 0
        for ind in range(0, len(other_images_data), len(other_images_data) // 10 + 1):  # step 1 (1-9), 2 (10-19) etc.
            other_data = other_images_data[ind]
            other_image = other_data["image"]
            is_same = self.reid.compare(person_image, other_image)
            same += is_same
            count += 1
        return same / count * 100

    @staticmethod
    def save_frame(frame: np.ndarray) -> None:
        """ Save frame to recordings directory. """
        file_name = f"{get_date_now_formatted()}.jpg"
        cv2.imwrite(f"{REC_PATH}/{file_name}", frame)


if __name__ == "__main__":
    main = MainClass()
    main.file_capturing()
