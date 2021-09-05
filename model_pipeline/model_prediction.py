"""
Python script for model prediction pipeline
"""
import tensorflow as tf
import numpy as np


class ModelPredictPipe:

    def __init__(self):
        """
        Function to initialize the class
        """
        self.variables = dict()

    @staticmethod
    def do_prediction(**kwargs):
        """
        Function to predict on the image
        :param kwargs:
        :return:
        """
        img_path = tf.keras.utils.get_file('store/imgs/predict_img.png',
                                           origin=kwargs['image_uri'])

        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=(kwargs['img_size'], kwargs['img_size'])
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = kwargs['model'].predict(img_array)
        score = predictions[0]

        outputs = {
            'predicted_label': kwargs['class_names'][np.argmax(score)],
            'predicted_score': 100 * np.argmax(score)
        }

        return outputs

    def get_activation_maps(self):
        """
        Function to get the activation maps
        :return:
        """
        pass
