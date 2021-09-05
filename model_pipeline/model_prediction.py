"""
Python script for model prediction pipeline
"""
import tensorflow as tf
import numpy as np
import keract
import os


class ModelPredictPipe:

    def __init__(self):
        """
        Function to initialize the class
        """
        self.variables = dict()

    @staticmethod
    def download_file(img_uri, output_path):
        """
        Function to download the image from the internet
        :param img_uri:
        :param output_path:
        :return:
        """
        if not os.path.exists(output_path):
            import requests
            download_img = requests.get(img_uri)
            with open(output_path, 'wb') as f:
                f.write(download_img.content)

    @staticmethod
    def get_preprocess_model():
        """
        Function to get the preprocessing layer model
        :return:
        """
        norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1/255)
        preprocessing_model = tf.keras.Sequential([norm_layer])
        return preprocessing_model

    def do_prediction(self, **kwargs):
        """
        Function to predict on the image
        :param kwargs:
        :return:
        """
        output_path = 'store/imgs/predict_img.png'

        self.download_file(kwargs['img_uri'], output_path)

        # preprocess image

        img = tf.keras.preprocessing.image.load_img(output_path, target_size=kwargs['model_img_size'])
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        preprocess_model = self.get_preprocess_model()
        predict_data = preprocess_model(img_array)
        predictions = kwargs['model'].predict(predict_data)
        score = predictions[0]

        outputs = {
            'predicted_label': kwargs['class_names'][np.argmax(score)],
            'predicted_score': 100 * np.argmax(score)
        }

        if len(kwargs['activation_layer_name']) == 0:
            layer_name = None
        else:
            layer_name = kwargs['activation_layer_name']

        activations = keract.get_activations(kwargs['model'], predict_data, layer_names=layer_name)
        keract.display_activations(activations, save=True, directory='store/activation_maps/', data_format='channels_last')

        return outputs
