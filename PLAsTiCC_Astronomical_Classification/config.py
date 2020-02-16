import os

ROOT = "/home/neko/data_science/PLAsTiCC_Astronomical_Classification"

TRAIN_CSV = os.path.join(ROOT, "training_set.csv")
TRAIN_METADATA_CSV = os.path.join(ROOT, "training_set_metadata.csv")

TEST_CSV = os.path.join(ROOT, "test_set.csv")
TEST_METADATA_CSV = os.path.join(ROOT, "test_set_metadata.csv")

assert os.path.exists(TRAIN_CSV)
assert os.path.exists(TRAIN_METADATA_CSV)
assert os.path.exists(TEST_CSV)
assert os.path.exists(TEST_METADATA_CSV)
