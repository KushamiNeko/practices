# import subprocess
import os

LOCAL_FEATURES = "/run/media/onionhuang/HDD/ARTIFICIAL_INTELLIGENCE/KAGGLE_COMPETITIONS/Carvana_Image_Masking_Challenge/test"

split = os.path.split(LOCAL_FEATURES)

FILE_LIST = os.path.join(split[0], "{}_list.txt".format(split[1]))

with open(FILE_LIST, "w") as f:
  files = [os.path.join(LOCAL_FEATURES, x) for x in os.listdir(LOCAL_FEATURES)]

  files_list = ",".join(files)

  print(files_list)

  print(len(files))

  f.write(files_list)

# JOB_DIR = "outputs"

# for i in range(1, 16):

# index = i + 1

# pattern = "*_{:02d}*".format(index)

# checkpoint = "outputs_{:02d}/checkpoint_3000.ckpt".format(index)

# command = [
# "python3", "main.py", "--features-dir", LOCAL_FEATURES, "--job-dir",
# JOB_DIR, "--pattern", pattern, "--checkpoint", checkpoint
# ]

# subprocess.check_output(command)
