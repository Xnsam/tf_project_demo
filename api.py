from fastapi import FastAPI, Request
from pydantic import BaseModel
from base.base_util import BaseUtil
from fastapi.templating import Jinja2Templates


app = FastAPI()
base_obj = BaseUtil()
template_obj = Jinja2Templates(directory="templates")
variables = dict()


class DataAPIInput(BaseModel):
    src: str
    data_uri: str
    model_name: str
    activation_map_layer: str


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
    variables['user_input_dict'] = input_dict
    result = {"src": input_dict.src, "data_uri": input_dict.data_uri}
    response = {"status": False, "user_msg": "Data Download Failed"}
    if base_obj.load_data(**result):
        response['status'] = True
        response['user_msg'] = "Data Download Complete"

    return response






