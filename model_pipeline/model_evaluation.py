"""
Python Script to evaluate the model
"""


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
        loss_, accuracy_ = kwargs['model'].evaluate(kwargs['test_data'], verbose=0)
        return loss_, accuracy_

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
        classification_report = dict()
        return classification_report

    def run_eval_pipeline(self, **kwargs):
        """
        Function to run the evaluation pipeline
        :param kwargs:
        :return:
        """
        report = dict()

        report['loss'], report['accuracy'] = self.get_loss_acc(**kwargs)
        report['classification_report'] = self.get_classification_report(**kwargs)



