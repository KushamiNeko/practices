import mxnet as mx
from mxnet import nd, autograd

import os
import sys
import subprocess

import config
import preprocess
import helper
import net

###############################################################################

if __name__ == "__main__":

    dictionary = preprocess.LabelDictionary()

    files = os.listdir(config.TRAIN_CSV_FILES)

    for f in files:

        if os.path.exists(
                os.path.join(
                    "check_points/separate_training", "{}_1.params".format(
                        dictionary.get_label_from_filename(f)))):
            continue

        conv_net = net.ConvNet()

        for e in range(config.EPOCHCS):

            generator = helper.generator(
                os.path.join(config.TRAIN_CSV_FILES, f), dictionary)

            print(
                "============================================================")

            print("label: %s, epochs: %d" %
                  (dictionary.get_label_from_filename(f), e))

            print(
                "============================================================")

            print()

            looping = True
            while looping:
                imgs = []
                labels = []

                for _ in range(config.BATCH_SIZE):
                    try:
                        x, y = next(generator)
                        imgs.append(x)
                        # labels.append(y)
                        labels.append(0)
                    except StopIteration:
                        looping = False
                        break

                if len(imgs) == 0 or len(labels) == 0:
                    looping = False
                    break

                nd_data = nd.array(imgs)
                nd_labels = nd.array(labels)

                nd_data = nd.transpose(nd_data, (0, 3, 1, 2))

                nd_data = nd_data.as_in_context(config.CTX)
                nd_labels = nd_labels.as_in_context(config.CTX)

                with autograd.record():
                    output = conv_net.net(nd_data)
                    loss = conv_net.loss(output, nd_labels)

                loss.backward()
                conv_net.trainer.step(nd_data.shape[0])

                print("Loss: {}".format(nd.mean(loss).asscalar()))

            conv_net.net.save_parameters(
                "check_points/separate_training/{}_{}.params".format(
                    dictionary.get_label_from_filename(f), e))

        # break
        program = os.path.join("/home/neko/programming_projects",
                               "ARTIFICIAL_INTELLIGENCE_DATA_SCIENCE",
                               "Quick_Draw_Doodle_Recognition_Challenge",
                               "engine/main.py")

        assert (os.path.exists(program))

        subprocess.Popen(["python3", program])
        sys.exit(0)

###############################################################################

# def train_all():
#
#    dictionary = preprocess.LabelDictionary()
#
#    conv_net = net.ConvNet()
#
#    for e in range(config.EPOCHCS):
#
#        print(
#            "================================================================")
#
#        print("epochs: %d" % (e))
#
#        print(
#            "================================================================")
#
#        print()
#
#        generator = helper.train_data_extract(dictionary)
#
#        looping = True
#        while looping:
#            imgs = []
#            labels = []
#
#            for _ in range(config.BATCH_SIZE):
#                try:
#                    x, y = next(generator)
#                    imgs.append(x)
#                    labels.append(y)
#                except StopIteration:
#                    looping = False
#                    break
#
#            if len(imgs) == 0 or len(labels) == 0:
#                break
#
#            data = nd.array(imgs)
#            labels = nd.array(labels)
#
#            data = nd.transpose(data, (0, 3, 1, 2))
#
#            data = data.as_in_context(config.CTX)
#            labels = labels.as_in_context(config.CTX)
#
#            with autograd.record():
#                output = conv_net.net(data)
#                loss = conv_net.softmax_cross_entropy(output, labels)
#
#            loss.backward()
#            conv_net.trainer.step(data.shape[0])
#
#            print("Loss: {}".format(nd.mean(loss).asscalar()))
#
#        conv_net.net.save_parameters(
#            "check_points/second_attemp/epochs_{}.params".format(e))

###############################################################################
