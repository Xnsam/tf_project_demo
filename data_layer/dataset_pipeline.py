"""
Python Script to build dataset
"""

from base.base_util import BaseUtil
import tensorflow as tf
import traceback


class DatasetPipe:

    def __init__(self, data_dir):
        """
        Function to initialize the Dataset pipe
        """
        self.variables = dict()
        self.variables['data_dir'] = data_dir
        self.variables['log_file_path'] = 'logs/data_layer/data_layer.log'
        self.dataset = dict()
        self.base_obj = BaseUtil()

    @staticmethod
    def get_preprocessing_model(**kwargs):
        """
        Function to build the preprocessing model
        :param kwargs: augment_data_flag: boolean: Flag for considering data augmentation
        :return: tf.keras.Sequential Model
        """
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1/255)
        preprocessing_model = tf.keras.Sequential([normalization_layer])

        if kwargs['augment_data_flag']:
            preprocessing_model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(30))
            preprocessing_model.add(tf.keras.layers.experimental.preprocessing.RandomTranslation(0, 0.2))
            preprocessing_model.add(tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0))
            preprocessing_model.add(tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.2))
            preprocessing_model.add(tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'))

        return preprocessing_model

    def get_dataset(self, **kwargs):
        """
        Function to build the dataset as per the dataset type
        :param kwargs: dataset_type: str: training / validation
        :param kwargs: image_size: tuple: Size of the images
        :param kwargs: batch_size: int: number of samples in the dataset
        :return: tf.data.Dataset generator
        """
        tmp_data = tf.keras.preprocessing.image_dataset_from_directory(
            "../" + self.variables['data_dir'],
            validation_split=0.20,
            subset=kwargs['dataset_type'],
            label_mode="categorical",
            seed=123,
            image_size=kwargs['image_size'],
            batch_size=kwargs['batch_size']
        )
        self.variables['class_names'] = tuple(tmp_data.class_names)
        self.variables['{}_size'.format(kwargs['dataset_type'])] = tmp_data.cardinality().numpy()
        tmp_data = tmp_data.unbatch().batch(kwargs['batch_size'])
        # tmp_data = tmp_data.repeat()

        augment_data_flag = False
        if 'augment_data_flag' in kwargs:
            augment_data_flag = kwargs['augment_data_flag']

        preprocessing_model = self.get_preprocessing_model(augment_data_flag=augment_data_flag)
        tmp_data = tmp_data.map(lambda images, labels: (preprocessing_model(images), labels))

        return tmp_data

    def run_dataset_pipeline(self, **kwargs):
        """
        Function to run the build dataset pipeline
        :param kwargs: image_size:
        :param kwargs: batch_size:
        :param kwargs: augment_data_flag: boolean
        :return: status: boolean
        """
        stat = False
        try:

            # split into training and validation set
            for ds_type in ["training", "validation"]:
                data = self.get_dataset(dataset_type=ds_type, image_size=kwargs['image_size'],
                                        batch_size=kwargs['batch_size'])
                self.dataset[ds_type] = data

            # split into validation and test set
            val_batches = tf.data.experimental.cardinality(self.dataset['validation'])
            self.dataset['test'] = self.dataset['validation'].take(val_batches // 5)
            self.dataset['validation'] = self.dataset['validation'].skip(val_batches // 5)
            stat = True
        except Exception:
            self.base_obj.write_log(log_lvl="ERROR", file_name=self.variables['log_file_path'],
                                    error_desc=traceback.format_exc())
        return stat, "Dataset pipeline Run {}".format(stat)


