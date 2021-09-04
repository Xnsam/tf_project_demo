"""
Python Script to evaluate the model
"""

from sklearn.metrics import classification_report
import numpy as np


class ModelEvalPipe:

    def __init__(self):
        """
        Function to initialize the evaluation pipeline
        """
        self.variables = dict()

    @staticmethod
    def get_loss_acc(**kwargs):
        """
        Function to evaluate and return loss and accuracy
        :param kwargs:
        :return:
        """
        loss_, accuracy_ = kwargs['model'].evaluate(kwargs['test_data'],
                                                    batch_size=kwargs['batch_size'],
                                                    verbose=1)
        return round(loss_, 4), round(accuracy_, 4)

    @staticmethod
    def get_history_plot(**kwargs):
        """
        Function to get the history plot
        :param kwargs:
        :return:
        """
        pass

    @staticmethod
    def get_classification_report(**kwargs):
        """
        Function to calculate the classification report
        :param kwargs:
        :return:
        """
        results = kwargs['model'].predict(kwargs['test_data'],
                                          batch_size=kwargs['batch_size'])
        y_true = np.argmax(kwargs['y_true'], axis=-1)
        y_prediction = np.argmax(results, axis=-1)
        c_report = classification_report(y_true, y_prediction, output_dict=True)
        return c_report

    def run_eval_pipeline(self, **kwargs):
        """
        Function to run the evaluation pipeline
        :param kwargs:
        :return:
        """
        report = dict()

        report['loss'], report['accuracy'] = self.get_loss_acc(**kwargs)
        report['classification_report'] = self.get_classification_report(**kwargs)

        return report

