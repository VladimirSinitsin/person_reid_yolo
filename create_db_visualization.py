import cv2

from tqdm import tqdm
from pathlib import Path

import db

from config import ROOTPATH
from tools import reid_img_revert


def main():
    # Create directory.
    vis_path = Path.joinpath(ROOTPATH, "visualisation")
    vis_path.mkdir(exist_ok=True)

    # Read images from DB.
    all_images_data = db.select_all_images_data()
    for image_data in tqdm(all_images_data):
        write_image(image_data, vis_path=vis_path)


def write_image(image_data: dict, vis_path: Path) -> None:
    """ Convert and write image. """
    # Create person directory.
    image_path = Path.joinpath(vis_path, str(image_data["person_id"]))
    if not image_path.exists():
        image_path.mkdir()
    # Convert image to normal format and write it.
    image = reid_img_revert(image_data["image"])
    image_filename = f"{image_data['date']}_{image_data['prediction']}.png"
    cv2.imwrite(f"{image_path}/{image_filename}", image)


if __name__ == "__main__":
    main()
