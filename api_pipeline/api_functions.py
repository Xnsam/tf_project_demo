"""
Python script for API functions
"""
import numpy as np

from model_pipeline.model_evaluation import ModelEvalPipe
from model_pipeline.model_finetune import ModelPipe


class CustomTrainApi:

    def __init__(self):
        """
        Function to initialize the class
        """
        self.variables = dict()
        self.variables['train_status'] = False
        self.variables['reports'] = None

    def get_train_status(self):
        """
        Function to get the train status
        :return:
        """
        return self.variables['train_status']

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
        return reports

    def run_train_api(self, **kwargs):
        """
        Function to run the train api pipeline
        :param kwargs:
        :return:
        """
        outputs = self.start_train_model(**kwargs)
        import gc
        gc.collect()
        reports = self.get_eval_reports(**outputs)
        self.variables['reports'] = reports
        self.variables['train_status'] = True

    @staticmethod
    def start_train_model(**kwargs):
        """
        Function to run the dataset pipeline and run the model training
        :return:
        """
        model_pipeline = ModelPipe(model_name=kwargs['model_name'],
                                   fine_tune_flag=kwargs['fine_tune_flag'],
                                   data_dir=kwargs['data_dir'])

        _ = model_pipeline.run_model_pipeline()

        output = {
            'model': model_pipeline.variables['model'],
            'test': model_pipeline.dataset_obj.dataset['test'],
            'y_true': np.concatenate([y for x, y in model_pipeline.dataset_obj.dataset['test']], axis=0),
            'batch_size': model_pipeline.variables['batch_size'],
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

