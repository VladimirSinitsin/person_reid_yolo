from pathlib import Path

import numpy as np

ROOTPATH = Path(__file__).parent

# Reid image input shape.
REID_IMAGE_SHAPE = (1, 160, 60, 3)
REID_IMAGE_H = REID_IMAGE_SHAPE[1]
REID_IMAGE_W = REID_IMAGE_SHAPE[2]
REID_IMAGE_DTYPE = np.dtype("float64")

# DBMS debugging.
RECREATE_DB = True  # must False
