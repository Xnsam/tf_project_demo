"""
Python script to calculate the gradient based class activation maps
"""
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm


class GradCAM:

    def __init__(self):
        """
        Function to initialize the class
        """
        self.variables = dict()

    @staticmethod
    def get_img_array(img_path, size):
        """
        Function to load the image
        :return: array: numpy
        """
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
        array = tf.keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

    @staticmethod
    def make_grad_cam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        """
        Function to calculate the heat map using grad cam
        :return: heat map: array
        """
        # build grad model
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # compute gradients
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heat_map = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heat_map= tf.squeeze(heat_map)

        heat_map = tf.maximum(heat_map, 0) / tf.math.reduce_max(heat_map)
        return heat_map.numpy()

    def apply_gradcam(self, img_path, heatmap, alpha=0.1):
        """
        Function to apply grad cam
        :param img_path:
        :param heatmap:
        :param alpha:
        :return:
        """
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap('jet')

        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        super_imposed_img = jet_heatmap * alpha + img
        super_imposed_img = tf.keras.preprocessing.image.array_to_img(super_imposed_img)

        return super_imposed_img



