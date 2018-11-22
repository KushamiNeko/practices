import os

ROOT = os.path.join("/run/media/neko/HDD/ARTIFICIAL_INTELLIGENCE_DATA_SCIENCE",
                    "KAGGLE_COMPETITIONS")
COMPETITION = "PLAsTiCC_Astronomical_Classification/all"

TRAIN_CSV = os.path.join(ROOT, COMPETITION, "training_set.csv")
TRAIN_METADATA_CSV = os.path.join(ROOT, COMPETITION,
                                  "training_set_metadata.csv")

TEST_CSV = os.path.join(ROOT, COMPETITION, "test_set.csv")
TEST_SAMPLE_CSV = os.path.join(ROOT, COMPETITION, "test_set_sample.csv")
TEST_METADATA_CSV = os.path.join(ROOT, COMPETITION, "test_set_metadata.csv")

assert (os.path.exists(TRAIN_CSV))
assert (os.path.exists(TRAIN_METADATA_CSV))
assert (os.path.exists(TEST_CSV))
assert (os.path.exists(TEST_SAMPLE_CSV))
assert (os.path.exists(TEST_METADATA_CSV))