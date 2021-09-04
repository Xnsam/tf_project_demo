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

        self.model_pipeline = None
        self.eval_pipeline = None

    def get_train_status(self):
        """
        Function to get the train status
        :return:
        """
        return self.variables['train_status']

    def start_train_model(self, **kwargs):
        """
        Function to run the dataset pipeline and run the model training
        :return:
        """
        self.model_pipeline = ModelPipe(model_name=kwargs['model_name'],
                                        fine_tune_flag=kwargs['fine_tune_flag'],
                                        data_dir=kwargs['data_dir'])

        stat, reason = self.model_pipeline.run_model_pipeline()

        if stat:
            print("Model pipeline complete")
            # evaluate the model to get predictions
            self.eval_pipeline = ModelEvalPipe()
            self.variables['reports'] = self.eval_pipeline.run_eval_pipeline(
                model=self.model_pipeline.variables['model'],
                test_data=self.model_pipeline.dataset_obj.dataset['test'],
                y_true=np.concatenate([y for x, y in self.model_pipeline.dataset_obj.dataset['test']], axis=0),
                batch_size=self.model_pipeline.variables['batch_size']
            )

        if self.variables['reports']:
            print('completed evaluation')
            self.variables['train_status'] = True

        print(self.variables['train_status'])
        print(self.variables['reports'])


train_obj = CustomTrainApi()
inputs = {
    'model_name': "efficientnetv2-b0",
    'fine_tune_flag': False,
    'data_dir': 'store/data/covid-wwo-pneumonia-chest-xray/Data'
}
train_obj.start_train_model(**inputs)

