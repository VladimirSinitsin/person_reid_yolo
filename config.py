from pathlib import Path

import numpy as np

ROOTPATH = Path(__file__).parent

# DBMS debugging.
RECREATE_DB = False  # must False

# Reid image input shape.
REID_IMAGE_SHAPE = (1, 160, 60, 3)
REID_IMAGE_H = REID_IMAGE_SHAPE[1]
REID_IMAGE_W = REID_IMAGE_SHAPE[2]
REID_IMAGE_DTYPE = np.dtype("float64")

# Image enlargement with the addition of black margins until Reid.
BLACK_MARGINS = True
