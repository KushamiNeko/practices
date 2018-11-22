import os

import mxnet as mx

###############################################################################

ROOT: str = os.path.join("/run/media/neko/HDD",
                         "ARTIFICIAL_INTELLIGENCE_DATA_SCIENCE",
                         "KAGGLE_COMPETITIONS")
COMPETIION: str = "Quick_Draw_Doodle_Recognition_Challenge"

TRAIN_CSV_FILES: str = os.path.join(ROOT, COMPETIION, "all/train_simplified")
TEST_CSV_FILE: str = os.path.join(ROOT, COMPETIION, "all/test_simplified.csv")

SIZE: int = 256

RESIZE: int = 64

BATCH_SIZE: int = 256

TRAIN_SAMPLES_LIMIT: int = 32

EPOCHCS: int = 500

###############################################################################

CTX = mx.gpu()

###############################################################################

assert (os.path.exists(ROOT))
assert (os.path.exists(TRAIN_CSV_FILES))
assert (os.path.exists(TEST_CSV_FILE))

###############################################################################
