"""
Python script to define class containing basic utility functions
"""
import traceback
import os
from kaggle.api.kaggle_api_extended import KaggleApi


class BaseUtil:

    def __init__(self):
        """
        Function to initialize the class
        """
        self.variables = dict()
        self.variables["data_store"] = "store/data"

    @staticmethod
    def data_exists(file_path):
        """
        Function to check if the data already exists
        :param file_path:
        :return:
        """
        return True if os.path.exists(file_path) else False

    def load_data(self, **kwargs):
        """
        Function to load data from multiple sources
        :param kwargs: src: str: Source of the data to pulled from
        :param kwargs: data_uri: str: Link of the data to pull from

        :return:
        """
        stat = False
        try:
            if kwargs['src'] == 'kaggle':  # function to download data from kaggle
                file_path = kwargs['data_uri'].split("/")[-1]
                file_path = self.variables['data_store'] + "/" + file_path
                print(file_path)
                if not self.data_exists(file_path):
                    os.mkdir(file_path)
                    api = KaggleApi()
                    api.authenticate()
                    api.dataset_download_files(kwargs['data_uri'], unzip=True, path=file_path)
                    stat = True
        except Exception:
            print(traceback.format_exc())

        return stat

