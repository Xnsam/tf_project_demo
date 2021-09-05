"""
Python script to define model fine tune class
"""
import os
import traceback
from data_layer.dataset_pipeline import DatasetPipe
import tensorflow as tf
import tensorflow_hub as tf_hub
import onnxruntime as onx_rt
import tf2onnx


class ModelPipe:

    def __init__(self, **kwargs):
        """
        Function to initialize the model fine tune pipeline
        :param kwargs: model_name: str: Name of the model
        :param kwargs: fine_tune_flag: boolean: To consider to fine tune or not
        :param kwargs: data_dir: str: Directory path for data

        """
        self.variables = dict()
        self.variables['model_name'] = kwargs['model_name']
        self.variables['log_file_path'] = 'logs/model_pipeline/model_pipeline.log'
        self.variables['model_save_path'] = 'store/model/{}'.format(self.variables['model_name'])
        self.variables['fine_tune_flag'] = kwargs['fine_tune_flag']
        self.variables['fine_tune_lyr'] = kwargs['fine_tune_lyr']
        self.variables['model_image_size'] = tuple()
        self.variables['batch_size'] = 16
        self.variables['model'] = None
        self.dataset_obj = DatasetPipe(kwargs['data_dir'])
        self.base_obj = self.dataset_obj.base_obj

    def convert_to_onnx(self):
        """
        Function to convert the model into ONNX format
        :return:
        """
        stat = True
        try:
            pixel_size = self.variables['model_image_size'][0]
            spec = (tf.TensorSpec((None, pixel_size, pixel_size, 3), tf.float32, name="input"),)
            output_path = self.variables['model'].name + ".onnx"

            model_proto, _ = tf2onnx.convert.from_keras(
                self.variables['model'], input_signature=spec, opset=13, output_path=output_path
            )
            self.variables['output_names'] = [n.name for n in model_proto.graph.output]
        except Exception:
            self.base_obj.write_log(file_name=self.variables['log_file_path'], log_lvl="ERROR",
                                    error_desc=traceback.format_exc())
        return stat, "Convert To ONNX {}".format(stat)

    def get_model(self):
        """
        Function to get the model
        :return:
        """
        base_model = None
        if self.variables['model_name'] == 'MobileNetV2':
            base_model = tf.keras.applications.MobileNetV2()
        if self.variables['model_name'] == "VGG16":
            base_model = tf.keras.applications.VGG16()

        return base_model

    def create_model(self):
        """
        Function to create model
        :return:
        """
        stat = False
        try:
            # build the model
            base_model = self.get_model()
            model = tf.keras.Sequential()

            for layer in base_model.layers[:-1]:
                layer.trainable = False
                model.add(layer)

            # Fine tuning the model
            if self.variables['fine_tune_flag']:
                assert self.variables['fine_tune_lyr'] < len(base_model.layers)
                for layer in model.layers[-self.variables['fine_tune_lyr']:]:
                    layer.trainable = True

            model.add(tf.keras.layers.Dense(len(self.dataset_obj.variables['class_names']),
                                            activation="softmax"))

            model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=['accuracy'])

            self.variables['model'] = model
            stat = True
        except Exception:
            self.base_obj.write_log(file_name=self.variables['log_file_path'], log_lvl="ERROR",
                                    error_desc=traceback.format_exc())
        return stat, "Create Model {}".format(str(stat))

    def get_model_details(self):
        """
        Function to get the model details
        :return: status: boolean
        """
        stat = False
        try:
            self.variables['model_uri'] = self.base_obj.variables['config']['model_pipeline']['model_name_map'].get(
                self.variables['model_name']
            )
            pixel_size = self.base_obj.variables['config']['model_pipeline']['model_size_map'].get(
                self.variables['model_name']
            )
            self.variables['model_image_size'] = (pixel_size, pixel_size)
            stat = True
        except Exception:
            self.base_obj.write_log(file_name=self.variables['log_file_path'], log_lvl="ERROR",
                                    error_desc=traceback.format_exc())

        return stat, "Get Model Details {}".format(str(stat))

    def load_model(self):
        """
        Function to load the model
        :return:
        """
        stat = False
        try:
            stat, reason = self.create_model()
            if stat:
                print(reason)
                self.variables['model'].load_weights(self.variables['model_path'])
                stat = True
        except Exception:
            self.base_obj.write_log(file_name=self.variables['log_file_path'], log_lvl="ERROR",
                                    error_desc=traceback.format_exc())
        return stat, "Load Model {}".format(stat)

    def run_onnx_model(self):
        """
        Function to run the ONNX model
        :return:
        """
        stat = False
        try:
            pass
        except Exception:
            pass
        return stat, "Run ONNX model {}".format(stat)

    def run_model_pipeline(self):
        """
        Function to run the pipeline
        :return:
        """

        # download model
        stat, reason = self.get_model_details()

        # load the dataset
        if stat:
            print(reason)
            stat, reason = self.dataset_obj.run_dataset_pipeline(
                image_size=self.variables['model_image_size'], batch_size=self.variables['batch_size'],
                augment_data_flag=False
            )

        # create custom model
        if stat:
            print(reason)
            stat, reason = self.create_model()

        # fine tune model on data
        if stat:
            print(reason)
            stat, reason = self.train_model()

        # save fine tuned model
        if stat:
            print(reason)
            stat, reason = self.save_model()

        # load fine tuned model
        # if stat:
        #     print(reason)
        #     stat, reason = self.load_model()

        else:
            print(reason)
            self.base_obj.write_log(file_name=self.variables['log_file_path'], log_lvl='ERROR', error_desc=reason)

        return stat, reason

    def save_model(self):
        """
        Function to save the keras fine tuned model
        :return: status: boolean
        """
        stat = False
        try:
            if not os.path.exists(self.variables['model_save_path']):
                os.mkdir(self.variables['model_save_path'])
            self.variables['model_path'] = "{}/fine_tuned_model".format(self.variables['model_save_path'])
            self.variables['model'].save_weights(self.variables['model_path'])
            stat = True
        except Exception:
            self.base_obj.write_log(log_lvl="ERROR", error_desc=traceback.format_exc(),
                                    file_name=self.variables['log_file_path'])
        return stat, "Save Model {}".format(stat)

    def train_model(self):
        """
        Function to train the model
        :return: status: boolean
        """
        stat = False
        try:
            steps_per_epoch = self.dataset_obj.variables['training_size'] // self.variables['batch_size']
            validation_steps = self.dataset_obj.variables['validation_size'] // self.variables['batch_size']

            # callbacks_list = [
            #     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
            #     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
            # ]

            self.variables['model_history'] = self.variables['model'].fit(
                self.dataset_obj.dataset['training'],
                epochs=5,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.dataset_obj.dataset['validation'],
                validation_steps=validation_steps
            ).history
            stat = True
        except Exception:
            self.base_obj.write_log(file_name=self.variables['log_file_path'], log_lvl="ERROR",
                                    error_desc=traceback.format_exc())
        return stat, "Train Model {}".format(str(stat))
