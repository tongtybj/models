#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
from glob import glob
import re
import argparse
import collections
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import datetime

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '../..'))


class LstmSSD():
    def __init__(self, model_filepath):

        print('Loading frozen graphmodel...')
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # checkout from protobug
        placeholder_name = None
        for n in graph_def.node:
            if  ('detection') in n.name:
                print("detection: {}".format(n.name))
            if n.op in ('Placeholder'):
                print("placeholder: {}".format(n.name))
                placeholder_name = n.name

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")
        self.graph.finalize()

        # check from tensorflow.graph
        input_tensor = self.graph.get_tensor_by_name(placeholder_name + ":0")
        print("input size: {}".format(input_tensor.shape))
        self._image_size = (input_tensor.shape[1], input_tensor.shape[2])
        self._sequence_size = input_tensor.shape[0]

        detection_boxes = self.graph.get_tensor_by_name("detection_boxes:0")

        print("output size: {}".format(detection_boxes.shape))

        self.sess = tf.Session(graph = self.graph)

    def get_image_size(self):
        return self._image_size

    def get_sequence_size(self):
        return self._sequence_size

    def detection(self, input_images):

        detection_classes = self.graph.get_tensor_by_name("detection_classes:0")
        detection_boxes = self.graph.get_tensor_by_name("detection_boxes:0")
        detection_scores = self.graph.get_tensor_by_name("detection_scores:0")

        tensor_classes, tensor_boxes, tensor_scores = self.sess.run([detection_classes, detection_boxes, detection_scores], feed_dict = {"image_tensor:0": input_images})

        #print(tensor_classes.shape)
        #print(tensor_boxes.shape)
        #print(tensor_scores.shape)

        #print(tensor_classes[3][0])
        #print(tensor_scores[3][0])
        #print(tensor_boxes[3][0])

        return tensor_classes, tensor_boxes, tensor_scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', dest='model_filepath', action="store",
                            help='the path of frozen graph model for inference', default='inference_model/frozen_graph.pb', type=str)

    parser.add_argument('--images', dest='image_filepath', action="store",
                            help='the path of iamges to do inference', default='assets/drone', type=str)
    args, _ = parser.parse_known_args()

    lstm_ssd = LstmSSD(args.model_filepath)
    filenames = sorted(glob(args.image_filepath + '/*.JPEG'))
    #print("image filenames: {}".format(filenames))

    first_image = cv2.imread(filenames[0])
    print(first_image.shape)
    if first_image.shape[0] != lstm_ssd.get_image_size()[0] or first_image.shape[1] != lstm_ssd.get_image_size()[1]:
        raise ValueError("the inference input size and image size are not consitent {} vs {}".format(first_image.shape, lstm_ssd.get_iamge_size()))


    for i, filename in enumerate(filenames):
        if i + lstm_ssd.get_sequence_size() < len(filenames):
            input_images = []
            for j in range(lstm_ssd.get_sequence_size()):
                input_images.append(cv2.imread(filenames[i + j]))
            input_images = np.stack(input_images)
            #print (input_images.shape)

            tensor_classes, tensor_boxes, tensor_scores = lstm_ssd.detection(input_images)

            # visualize
            for j in range(lstm_ssd.get_sequence_size()):
                min_y = int(tensor_boxes[j][0][0] * lstm_ssd.get_image_size()[0])
                max_y = int(tensor_boxes[j][0][2] * lstm_ssd.get_image_size()[0])
                min_x = int(tensor_boxes[j][0][1] * lstm_ssd.get_image_size()[1])
                max_x = int(tensor_boxes[j][0][3] * lstm_ssd.get_image_size()[1])

                detection_image = cv2.rectangle(input_images[j], (min_x, min_y), (max_x, max_y), (0,255,0),2)

                cv2.imshow('deteciont' + str(j + 1), detection_image)
            k = cv2.waitKey(0)
            if k == 27:         # wait for ESC key to exit
                sys.exit()

            #raise ValueError("OK")
