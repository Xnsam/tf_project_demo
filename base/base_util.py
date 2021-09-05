"""
Python script to define class containing basic utility functions
"""
import traceback
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import yaml
import datetime
import requests


class BaseUtil:

    def __init__(self):
        """
        Function to initialize the class
        """
        self.variables = dict()
        self.variables['folder_paths'] = [
            "store",
            "store/data",
            "store/model",
            "store/plots",
            "store/imgs",
            "logs",
            "logs/model_pipeline",
            "logs/utilities",
            "logs/data_layer"
        ]

        self.variables["config_file_path_1"] = "../config.yaml"
        self.variables["config_file_path_2"] = "config.yaml"
        self.variables["log_file_path"] = "logs/utilities/base_util.log"
        self.variables['data_store'] = "store/data"
        self.run_adhoc()

    def create_folders(self):
        """
        Function to create missing folders
        :return:
        """
        for folder_path in self.variables['folder_paths']:
            if not self.data_exists(folder_path):
                folder_path = '../' + folder_path
            if not self.data_exists(folder_path):
                os.mkdir(folder_path)
                print('Created ' + folder_path)

    @staticmethod
    def data_exists(file_path):
        """
        Function to check if the data already exists
        :param file_path: str: filepath to check
        :return: Status
        """
        return True if os.path.exists(file_path) else False

    def load_data(self, **kwargs):
        """
        Function to load data from multiple sources
        :param kwargs: src: str: Source of the data to pulled from
        :param kwargs: data_uri: str: Link of the data to pull from

        :return: status: boolean
        """
        stat = False
        try:
            if kwargs['src'] == 'kaggle':  # function to download data from kaggle
                file_path = kwargs['data_uri'].split("/")[-1]
                file_path = self.variables['data_store'] + "/" + file_path
                if not self.data_exists(file_path):
                    os.mkdir(file_path)
                    api = KaggleApi()
                    api.authenticate()
                    api.dataset_download_files(kwargs['data_uri'], unzip=True, path=file_path)
                self.variables['data_dir'] = file_path + '/' + 'Data'
                stat = True
        except Exception:
            self.write_log(file_name=self.variables['log_file_path'], log_lvl="ERROR",
                           error_desc=traceback.format_exc())

        return stat

    def load_config_file(self):
        """
        Function to load the config file
        :return:
        """
        if os.path.exists(self.variables['config_file_path_1']):
            with open(self.variables['config_file_path_1'], 'r') as f:
                self.variables['config'] = yaml.load(f, Loader=yaml.Loader)
        elif os.path.exists(self.variables['config_file_path_2']):
            with open(self.variables['config_file_path_2'], 'r') as f:
                self.variables['config'] = yaml.load(f, Loader=yaml.Loader)
        else:
            raise FileNotFoundError

    def run_adhoc(self):
        """
        Function to run the adhoc methods
        :return: None
        """

        self.load_config_file()
        self.create_folders()

    @staticmethod
    def write_log(file_name, log_lvl, error_desc):
        """
        Function to write the log
        :param file_name: str: Name of the file to write
        :param log_lvl: str: Level of the log DEBUG / ERROR / WARN / CRITICAL
        :param error_desc: str: Description of the log file
        :return: None
        """
        time_stamp = datetime.datetime.now().strftime("%D %H:%M:%S")
        with open(file_name, 'a') as f:
            log_line = "[{}]:[{}]:[{}]".format(time_stamp, log_lvl, error_desc)
            f.write(log_line + "\n")

