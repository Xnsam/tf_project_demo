"""
Python script to define model fine tune class
"""
import traceback
from data_layer.dataset_pipeline import DatasetPipe
import tensorflow as tf
import tensorflow_hub as tf_hub


class ModelPipe:

    def __init__(self, **kwargs):
        """
        Function to initialize the model fine tune pipeline
        :param kwargs: model_name: str: Name of the model
        :param kwargs: fine_tune_flag: boolean: To consider to fine tune or not
        :param kwargs: data_dir: str: Directory path for data

        """
        self.variables = dict()
        self.variables['model_store'] = 'store/model'
        self.variables['model_name'] = kwargs['model_name']
        self.variables['log_file_path'] = 'logs/model_pipeline/model_pipeline.log'
        self.variables['fine_tune_flag'] = kwargs['fine_tune_flag']
        self.variables['model_image_size'] = tuple()
        self.model = None
        self.dataset_obj = DatasetPipe(kwargs['data_dir'])
        self.base_obj = self.dataset_obj.base_obj

    def create_model(self):
        """
        Function to create model
        :return:
        """
        stat = False
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=self.variables['model_image_size'] + (3,)),
                tf_hub.KerasLayer(self.variables['model_uri'], trainable=self.variables['fine_tune_flag']),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(len(self.dataset_obj.variables['class_names']),
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001))
            ])

            model.build((None, ) + self.variables['model_image_size'] + (3,))
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                metrics=['accuracy']
            )
            self.model = model
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
            self.variables['model_uri'] = self.base_obj.variables['config']['model_name_map'].get(
                self.variables['model_name']
            )
            pixel_size = self.base_obj.variables['config']['model_size_map'].get(
                self.variables['model_name']
            )
            self.variables['model_image_size'] = (pixel_size, pixel_size)
            stat = True
        except Exception:
            self.base_obj.write_log(file_name=self.variables['log_file_path'], log_lvl="ERROR",
                                    error_desc=traceback.format_exc())

        return stat, "Get Model Details {}".format(str(stat))

    def run(self):
        """
        Function to run the pipeline
        :return:
        """

        # download model
        stat, reason = self.get_model_details()

        # load the dataset
        if stat:
            stat, reason = self.dataset_obj.run(
                image_size=self.variables['model_image_size'], batch_size=16, augment_data_flag=False
            )

        # create custom model
        if stat:
            stat, reason = self.create_model()

        # fine tune model on data
        if stat:
            stat, reason = self.train_model()

        # save fine tuned model
        if stat:
            stat, reason = self.save_model()

        else:
            self.base_obj.write_log(file_name=self.variables['log_file_path'], log_lvl='ERROR', error_desc=reason)

        return stat

    def train_model(self):
        """
        Function to train the model
        :return: status: boolean
        """
        stat = False
        try:
            steps_per_epoch = self.variables['train_size'] // self.variables['batch_size']
            validation_steps = self.variables['valid_size'] // self.variables['batch_size']

            self.variables['model_history'] = self.model.fit(
                self.dataset_obj.dataset['training'],
                epochs=5,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.dataset_obj.dataset['validation'],
                validation_steps=validation_steps
            ).history()
        except Exception:
            self.base_obj.write_log(file_name=self.variables['log_file_path'], log_lvl="ERROR",
                                    error_desc=traceback.format_exc())
        return stat, "Train Model {}".format(str(stat))
