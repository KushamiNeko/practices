import os
import pickle

import config
import helper

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
            # label = self.get_label_from_filename(f)
            label = helper.get_label_from_filepath(f)

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

    # def get_label_from_filename(self, filename: str) -> str:
    # label = os.path.splitext(os.path.basename(filename))[0].replace(
    # " ", "_")
    # return label


###############################################################################
