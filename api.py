from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from base.base_util import BaseUtil
from api_pipeline.api_functions import CustomApi

import warnings
warnings.filterwarnings("ignore")


variables = dict()

app = FastAPI()
base_obj = BaseUtil()
template_obj = Jinja2Templates(directory="templates")
api_obj = CustomApi()


class DataAPIInput(BaseModel):
    src: str
    data_uri: str


class ModelTrainAPIInput(BaseModel):
    model_name: str
    fine_tune_flag: bool
    fine_tune_lyr: int


class ModelPredictAPIInput(BaseModel):
    image_uri: str
    activation_layer_name: list


@app.get("/")
async def root():
    """
    Demo function to check the connectivity
    :return: Dict: Hello world message
    """
    return {"message": "Hello World"}


@app.get("/demo")
def demo(request: Request):
    """
    API for serving demo page
    :return: TemplateResponse HTML
    """
    return template_obj.TemplateResponse("dashboard.html", {
        "request": request,
        "some_value": 2
    })


@app.post("/fetch_data")
def data_api(input_dict: DataAPIInput):
    """
    API to fetch the data from the given source
    :param input_dict:
    :return:
    """
    result = {"src": input_dict.src, "data_uri": input_dict.data_uri}
    response = {"status": False, "user_msg": "Data Download Failed"}
    if base_obj.load_data(**result):
        response['status'] = True
        response['user_msg'] = "Data Download Complete"

    return response


@app.post("/model_train")
def model_train(input_dict: ModelTrainAPIInput, bg_tasks: BackgroundTasks):
    """
    API to train the given model
    :param bg_tasks:
    :param input_dict:
    :return:
    """
    if api_obj.variables['train_status'] is False:
        bg_tasks.add_task(api_obj.run_train_api, model_name=input_dict.model_name,
                          fine_tune_flag=input_dict.fine_tune_flag, fine_tune_lyr=input_dict.fine_tune_lyr,
                          data_dir=base_obj.variables['data_dir'])
        response = {"user_msg": "Model training initiated"}
    elif api_obj.variables['train_status'] == 'IP':
        response = {"user_msg": "Model training in progress"}
    else:
        response = {'evaluation_reports': api_obj.variables['reports'],
                    "user_msg": 'Model training complete'}
    return response


@app.post("/model_predict")
def model_predict(input_dict: ModelPredictAPIInput):
    """
    API to predict on the given data and return activation maps
    :param input_dict:
    :return:
    """
    api_obj.run_pred_api(model_path='D:/projects/tf_demo/tf_project_demo/store/model/VGG16/fine_tuned_model.hdf5',
                         img_uri=input_dict.image_uri,
                         class_names=api_obj.variables['class_names'],
                         activation_layer_name=input_dict.activation_layer_name)
    response = api_obj.prediction
    response['predicted_score'] = response['predicted_score'].tolist()
    output_path = response['validation_img']
    del response['validation_img']

    return FileResponse(output_path), response


if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=5000)
