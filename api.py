from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates

from base.base_util import BaseUtil
from api_pipeline.api_functions import start_train_model


variables = dict()

app = FastAPI()
base_obj = BaseUtil()
template_obj = Jinja2Templates(directory="templates")


class DataAPIInput(BaseModel):
    src: str
    data_uri: str


class ModelTrainAPIInput(BaseModel):
    model_name: str
    fine_tune_flag: bool


class ModelPredictAPIInput(BaseModel):
    image_uri: str


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
async def model_train(input_dict: ModelTrainAPIInput,
                      bg_tasks: BackgroundTasks):
    """
    API to train the given model
    :param bg_tasks:
    :param input_dict:
    :return:
    """
    bg_tasks.add_task(start_train_model, model_name=input_dict.model_name,
                      fine_tune_flag=input_dict.fine_tune_flag, data_dir=base_obj.variables['data_dir'])
    response = {"user_msg": "Model training initiated"}
    return response


@app.post("/model_predict")
def model_predict(input_dict: ModelPredictAPIInput):
    """
    API to predict on the given data and return activation maps
    :param input_dict:
    :return:
    """
    pass
