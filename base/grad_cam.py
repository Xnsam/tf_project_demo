"""
Python script to calculate the gradient based class activation maps
"""

import tensorflow as tf
import numpy as np
import cv2


class GradCAM:

    def __init__(self, model, class_idx, layer_name=None):
        """
        Function to initialize the class
        """
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name

        if self.layer_name is None:
            self.layer_name = self.find_target_layer()

    def find_target_layer(self):
        """
        Function to return  
        :return:
        """