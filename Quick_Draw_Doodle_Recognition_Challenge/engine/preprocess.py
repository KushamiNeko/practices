import os
import pickle

import config

###############################################################################


class LabelDictionary():

    LABEL_ID_DICT: dict = {}
    ID_LABEL_DICT: dict = {}

    def __init__(self):
        self.__load_dictionary()

    def __load_dictionary(self):

        if (not os.path.exists("resource/label_id.pickle") or
                not os.path.exists("resource/id_label.pickle")):

            print("generate label dict")

            self.__generate_dictionary()

        else:
            label_id = open("resource/label_id.pickle", "rb")
            self.LABEL_ID_DICT = pickle.load(label_id)
            label_id.close()

            id_label = open("resource/id_label.pickle", "rb")
            self.ID_LABEL_DICT = pickle.load(id_label)
            id_label.close()

    def __generate_dictionary(self):

        for index, f in enumerate(os.listdir(config.TRAIN_CSV_FILES)):
            # label = os.path.splitext(f)[0].replace(" ", "_")
            label = self.get_label_from_filename(f)

            if label not in self.LABEL_ID_DICT:
                self.LABEL_ID_DICT[label] = index

            if index not in self.ID_LABEL_DICT:
                self.ID_LABEL_DICT[index] = label

        label_id = open("resource/label_id.pickle", 'wb')
        pickle.dump(obj=self.LABEL_ID_DICT, file=label_id)
        label_id.close()

        id_label = open("resource/id_label.pickle", 'wb')
        pickle.dump(obj=self.ID_LABEL_DICT, file=id_label)
        id_label.close()

    def get_label_from_index(self, index: int) -> str:
        return self.ID_LABEL_DICT[index]

    def get_index_from_label(self, label: str) -> int:
        return self.LABEL_ID_DICT[label]

    def get_label_from_filename(self, filename: str) -> str:
        label = os.path.splitext(os.path.basename(filename))[0].replace(
            " ", "_")
        return label


###############################################################################

# def get_label_index(filename: str) -> int:
# label = os.path.splitext(os.path.basename(filename))[0].replace(" ", "_")
# return LABEL_ID_DICT[label]

###############################################################################

# def get_index_label(index: int) -> str:
# return ID_LABEL_DICT[index]

###############################################################################

# def get_label_dict():

# global LABEL_ID_DICT
# global ID_LABEL_DICT

# if not os.path.exists("resource/label_id.pickle") or not os.path.exists(
# "resource/id_label.pickle"):

# print("generate label dict")

# generate_label_dict()

# else:
# label_id = open("resource/label_id.pickle", "rb")
# LABEL_ID_DICT = pickle.load(label_id)
# label_id.close()

# id_label = open("resource/id_label.pickle", "rb")
# ID_LABEL_DICT = pickle.load(id_label)
# id_label.close()

###############################################################################

# def generate_label_dict():

# for index, f in enumerate(os.listdir(config.TRAIN_CSV_FILES)):
# label = os.path.splitext(f)[0].replace(" ", "_")

# if label not in LABEL_ID_DICT:
# LABEL_ID_DICT[label] = index

# if index not in ID_LABEL_DICT:
# ID_LABEL_DICT[index] = label

# label_id = open("resource/label_id.pickle", 'wb')
# pickle.dump(obj=LABEL_ID_DICT, file=label_id)
# label_id.close()

# id_label = open("resource/id_label.pickle", 'wb')
# pickle.dump(obj=ID_LABEL_DICT, file=id_label)
# id_label.close()

###############################################################################

# def print_label_dict():
# label_id = open("resource/label_id.pickle", "rb")
# t = pickle.load(label_id)
# label_id.close()

# print(t)
# print(len(t))

# id_label = open("resource/id_label.pickle", "rb")
# t = pickle.load(id_label)
# id_label.close()

# print(t)
# print(len(t))
