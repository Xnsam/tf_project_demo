"""
Python script for API functions
"""
import numpy as np
from model_pipeline.model_evaluation import ModelEvalPipe
from model_pipeline.model_train import ModelPipe
from model_pipeline.model_prediction import ModelPredictPipe


class CustomApi:

    def __init__(self):
        """8
        Function to initialize the class
        """
        self.variables = dict()
        self.variables['train_status'] = False
        self.variables['reports'] = None
        self.base_obj = None
        self.prediction = dict()
        self.activation_maps = None

    @staticmethod
    def get_eval_reports(**kwargs):
        """
        Function to get the evaluation reports
        :return:
        """
        eval_pipeline = ModelEvalPipe()
        reports = eval_pipeline.run_eval_pipeline(
            model=kwargs['model'],
            test_data=kwargs['test'],
            y_true=kwargs['y_true'],
            batch_size=kwargs['batch_size']
        )
        print("Evaluate Model True")
        return reports

    def run_train_api(self, **kwargs):
        """
        Function to run the train api pipeline
        :param kwargs:
        :return:
        """
        self.variables['train_status'] = 'IP'
        outputs = self.start_train_model(**kwargs)
        reports = self.get_eval_reports(**outputs)
        self.variables['reports'] = reports
        self.variables['train_status'] = True
        self.variables['model'] = outputs['model']
        self.variables['batch_size'] = outputs['batch_size']
        self.variables['model_img_size'] = outputs['model_img_size']
        self.variables['class_names'] = outputs['class_names']

    def run_pred_api(self, **kwargs):
        """
        Function to run the prediction api pipeline
        :param kwargs:
        :return:
        """
        # do predictions
        pred_pipeline = ModelPredictPipe()
        kwargs.update(self.variables)
        outputs = pred_pipeline.do_prediction(**kwargs)
        self.prediction['predicted_label'] = outputs['predicted_label']
        self.prediction['predicted_score'] = outputs['predicted_score']
        self.prediction['validation_img'] = outputs['validation_img']

        # import os
        # self.activation_maps = os.listdir('store/activation_maps')[-1]

        print('Prediction pipeline complete')

    @staticmethod
    def start_train_model(**kwargs):
        """
        Function to run the dataset pipeline and run the model training
        :return:
        """

        model_pipeline = ModelPipe(model_name=kwargs['model_name'],
                                   fine_tune_flag=kwargs['fine_tune_flag'],
                                   data_dir=kwargs['data_dir'], fine_tune_lyr=kwargs['fine_tune_lyr'])

        _ = model_pipeline.run_model_pipeline()

        output = {
            'model': model_pipeline.variables['model'],
            'test': model_pipeline.dataset_obj.dataset['test'],
            'y_true': np.concatenate([y for x, y in model_pipeline.dataset_obj.dataset['test']], axis=0),
            'batch_size': model_pipeline.variables['batch_size'],
            'model_img_size': model_pipeline.variables['model_image_size'],
            'class_names': model_pipeline.dataset_obj.variables['class_names']
        }
        output.update(kwargs)
        return output


# train_obj = CustomTrainApi()
# inputs = {
#     'model_name': "efficientnetv2-b0",
#     'fine_tune_flag': False,
#     'data_dir': 'store/data/covid-wwo-pneumonia-chest-xray/Data'
# }
# train_obj.run_train_api(**inputs)

