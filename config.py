from pathlib import Path

import numpy as np

ROOTPATH = Path(__file__).parent

# DBMS debugging.
RECREATE_DB = True  # must be False

# Reid image input shape.
REID_IMAGE_SHAPE = (1, 160, 60, 3)  # from model architecture
REID_IMAGE_H = REID_IMAGE_SHAPE[1]
REID_IMAGE_W = REID_IMAGE_SHAPE[2]
REID_IMAGE_DTYPE = np.dtype("float64")

# Image enlargement with the addition of black margins until Reid.
BLACK_MARGINS = True

# Video capturing settings.
SOURCE_VIDEO_FILE_PATH = f"{ROOTPATH}/test_data/we_1.mp4"
SAVE_RECORDS = True
REC_PATH = f"{ROOTPATH}/recordings"
EACH_FRAME = 5
XY_THRESHOLD = 20
IOU_THRESHOLD = 20
MAX_IOU_THRESHOLD = 50
