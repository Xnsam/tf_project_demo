import requests

json_dict = {
    "src": "kaggle",
    "data_uri": "rashikrahmanpritom/covid-wwo-pneumonia-chest-xray",
    "model_name": "some_value",
    "activation_map_layer": "some_value"
}

variables = {
    "host_name": "http://localhost",
    "port_num": "5000",
    "end_point": "fetch_data",
}

url_ = "{}:{}/{}".format(variables["host_name"], variables["port_num"], variables["end_point"])
resp = requests.post(url_, json=json_dict)
# url = "{}:{}/{}".format(variables["host_name"], variables["port_num"], "/")
# resp = requests.get(url)
print(resp.content)
