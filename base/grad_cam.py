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
        Function to return the target layer
        :return:
        """
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer, Cannot Apply GradCAM")

    def compute_heatmap(self, image, eps=1e-8):
        """
        Function to compute the heat map from the gradient model
        :param image:
        :param eps:
        :return:
        """
        # build the grad model
        grad_model = tf.keras.Model(
            inputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )

        # record the operations for automatic differentiation
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = grad_model(inputs)
            loss = predictions[:, self.class_idx]

        # compute gradients
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError("Could not compute gradients")

        # compute guided gradients
        cast_conv_outputs = tf.cast(conv_outputs>0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        # compute average of the gradient values
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        # match input image size and output image size
        (w, h) = (image.shape[2], image.shape[1])
        heat_map = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap
        numer = heat_map - np.min(heat_map)
        denom = (heat_map.max() - heat_map.min()) + eps
        heat_map = numer / denom

        # scale and convert into unsigned int
        heat_map = (heat_map * 255).astype('uint8')

        return heat_map

    def overlay_heatmap(self, **kwargs):
        """
        Function to apply the color map to the heat and overlay with input image
        :param kwargs:
        :return:
        """
        color_map = cv2.COLORMAP_VIRIDIS
        if 'color_map' in kwargs:
            color_map = kwargs['color_map']

        heat_map = cv2.applyColorMap(kwargs['heat_map'], color_map)
        output = cv2.addWeighted(kwargs['image'], kwargs['alpha'], heat_map, 1 - kwargs['alpha'], 0)
        return (heat_map, output)




