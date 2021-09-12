"""
Python script for model prediction pipeline
"""
import tensorflow as tf
import numpy as np
import keract
import os
from tf_explain.core.grad_cam import GradCAM


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

    # @staticmethod
    # def get_activation_maps(**kwargs):
    #     """
    #     Function to extract the activation map
    #     :return:
    #     """
    #     import pdb; pdb.set_trace()
    #     # new approach
    #     explainer = GradCAM()
    #     img_inp = tf.keras.preprocessing.image.load_img(kwargs['img_path'], target_size=kwargs['img_size'])
    #     img_inp = tf.keras.preprocessing.image.img_to_array(img_inp)
    #     grid = explainer.explain(validation_data=([img_inp], None), model=kwargs['model'], layer_name='block5_conv3', class_index=1)
    #
    #     grid = explainer.explain(validation_data=([img_inp], None), model=kwargs['model'], layer_name=kwargs['layer_name'][0], class_index=1)
    #
    #     import matplotlib.cm as cm
    #
    #     grid = np.uint8(255 * grid)
    #     jet = cm.get_cmap('jet')
    #     jet_colors = jet(np.arange(256))[:, :3]
    #     jet_heatmap = jet_colors[grid]
    #
    #     jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    #     jet_heatmap = jet_heatmap.resize((img_inp.shape[1], img_inp.shape[0]))
    #     jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    #
    #     superimposed_img = jet_heatmap * kwargs['alpha'] + img_inp
    #     superimposed_img = tf.keras.image.array_to_img(superimposed_img)
    #
    #     superimposed_img.save(kwargs['output_path'])

        # older approach
        # from tf_explain.core.grad_cam import GradCAM
        # explainer = GradCAM()
        # img_ = tf.keras.preprocessing.image.load_img(output_path, target_size=kwargs['model_img_size'])
        # img_array_ = tf.keras.preprocessing.image.img_to_array(img_)
        # import pdb; pdb.set_trace()
        # tmp_data = ([img_array_], None)
        # grid = explainer.explain(tmp_data, kwargs['model'], class_index=np.argmax(score))
        #
        # activations = keract.get_activations(kwargs['model'], predict_data, layer_names=layer_name)
        # keract.display_activations(activations, save=True, directory='store/activation_maps/',
        # data_format='channels_last')
        # return superimposed_img


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
        img_path = 'store/imgs/predict_img.png'

        model = tf.keras.models.load_model(kwargs['model_path'])
        self.download_file(kwargs['img_uri'], img_path)

        # preprocess image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=kwargs['model_img_size'])
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # predict on results
        preprocess_model = self.get_preprocess_model()
        predict_data = preprocess_model(img_array)
        predictions = model.predict(predict_data)
        score = predictions[0]

        # use grad cam for validation image
        explainer = GradCAM()
        img_inp = tf.keras.preprocessing.image.load_img(img_path, target_size=kwargs['model_img_size'])
        img_inp = tf.keras.preprocessing.image.img_to_array(img_inp)
        grid = explainer.explain(validation_data=([img_inp], None), model=model,
                                 layer_name=kwargs['activation_layer_name'][0],
                                 class_index=np.argmax(score))
        alpha = 0.3
        validation_img = grid * alpha + img_inp
        validation_img = tf.keras.preprocessing.image.array_to_img(validation_img)

        output_path = 'store/activation_maps/{}_gradcam.png'.format(
            kwargs['activation_layer_name'][0]
        )
        validation_img.save(output_path)

        outputs = {'predicted_label': kwargs['class_names'][np.argmax(score)], 'predicted_score': score,
                   'validation_img': output_path}

        return outputs
