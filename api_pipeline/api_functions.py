"""
Python script for API functions
"""

from model_pipeline.model_evaluation import ModelEvalPipe
from model_pipeline.model_finetune import ModelPipe


def start_train_model(model_name: str, fine_tune_flag: bool, data_dir: str):
    """
    Function to run the dataset pipeline and run the model training
    :return:
    """
    reports = None
    model_pipeline = ModelPipe(model_name=model_name, fine_tune_flag=fine_tune_flag,
                               data_dir=data_dir)

    stat, reason = model_pipeline.run_model_pipeline()

    if stat:
        print("Model pipeline complete")
        # evaluate the model to get predictions
        eval_pipeline = ModelEvalPipe()
        reports = eval_pipeline.run_eval_pipeline(
            model=model_pipeline.variables['model'],
            test_data=model_pipeline.dataset_obj.dataset['test_data']
        )

    return stat, reason, reports


output = start_train_model(model_name="efficientnetv2-b0", fine_tune_flag=True,
                           data_dir='store/data/covid-wwo-pneumonia-chest-xray/Data')
print(output)

