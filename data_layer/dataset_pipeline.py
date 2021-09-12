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

        if kwargs['model_name'] == 'Xception':
            preprocess_input = tf.keras.applications.xception.preprocess_input
        if kwargs['model_name'] == 'MobileNetV2':
            preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        if kwargs['model_name'] == 'VGG16':
            preprocess_input = tf.keras.applications.vgg16.preprocess_input

        # preprocessing_model = tf.keras.Sequential([preprocess_input])
        #
        # if kwargs['augment_data_flag']:
        #     preprocessing_model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(30))
        #     preprocessing_model.add(tf.keras.layers.experimental.preprocessing.RandomTranslation(0, 0.2))
        #     preprocessing_model.add(tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0))
        #     preprocessing_model.add(tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.2))
        #     preprocessing_model.add(tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal'))

        return preprocess_input

    def get_dataset(self, **kwargs):
        """
        Function to build the dataset as per the dataset type
        :param kwargs: dataset_type: str: training / validation
        :param kwargs: image_size: tuple: Size of the images
        :param kwargs: batch_size: int: number of samples in the dataset
        :return: tf.data.Dataset generator
        """
        tmp_data = None
        if kwargs['dataset_type'] in ["training", "validation"]:
            tmp_data = tf.keras.preprocessing.image_dataset_from_directory(
                self.variables['data_dir'] + kwargs['sub_dir'],
                validation_split=0.20,
                subset=kwargs['dataset_type'],
                label_mode="categorical",
                seed=123,
                image_size=kwargs['image_size'],
                batch_size=kwargs['batch_size']
            )
        else:
            tmp_data = tf.keras.preprocessing.image_dataset_from_directory(
                self.variables['data_dir'] + kwargs['sub_dir'],
                label_mode="categorical",
                seed=123,
                image_size=kwargs['image_size'],
                batch_size=kwargs['batch_size']
            )
        self.variables['class_names'] = tuple(tmp_data.class_names)
        print(" class names ", self.variables['class_names'])
        self.variables['{}_size'.format(kwargs['dataset_type'])] = tmp_data.cardinality().numpy()
        tmp_data = tmp_data.unbatch().batch(kwargs['batch_size'])
        if kwargs['dataset_type'] in ["training", "validation"]:
            tmp_data = tmp_data.repeat()

        augment_data_flag = False
        if 'augment_data_flag' in kwargs:
            augment_data_flag = kwargs['augment_data_flag']

        preprocessing_model = self.get_preprocessing_model(augment_data_flag=augment_data_flag,
                                                           model_name=kwargs['model_name'])
        tmp_data = tmp_data.map(lambda images, labels: (preprocessing_model(images), labels))

        return tmp_data

    def run_dataset_pipeline(self, **kwargs):
        """
        Function to run the build dataset pipeline
        :param kwargs: image_size: tuple
        :param kwargs: batch_size: int
        :param kwargs: augment_data_flag: boolean
        :param kwargs: model_name: str
        :return: status: boolean
        """
        stat = False
        try:
            sub_dir_map = {
                "training": '/train',
                "validation": '/train',
                "test": '/test',
            }
            # split into training and validation set
            for ds_type in ["training", "validation", "test"]:
                data = self.get_dataset(dataset_type=ds_type, image_size=kwargs['image_size'],
                                        batch_size=kwargs['batch_size'],
                                        model_name=kwargs['model_name'],
                                        sub_dir=sub_dir_map[ds_type])
                self.dataset[ds_type] = data

            stat = True
        except Exception:
            self.base_obj.write_log(log_lvl="ERROR", file_name=self.variables['log_file_path'],
                                    error_desc=traceback.format_exc())
        return stat, "Dataset pipeline Run {}".format(stat)


