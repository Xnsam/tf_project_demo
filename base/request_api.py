import requests


def do_post_request(json_dict, host_name="http://localhost",
                    port_num="5000", end_point="fetch_data"):
    """
    Function to do the post request
    :param json_dict: API POST Request Contents
    :param host_name: Name of the host
    :param port_num: Port number of the host name
    :param end_point: API to hit
    :return: response status code
    """
    url_ = "{}:{}/{}".format(host_name, port_num, end_point)
    resp = requests.post(url_, json=json_dict)

    return resp.status_code, resp.content


# Sample POST Request for fetch data
print(" POST Request for fetch data ")
api_request = {
    "src": "kaggle",
    "data_uri": "rashikrahmanpritom/covid-wwo-pneumonia-chest-xray",
}
status, resp = do_post_request(json_dict=api_request)
if status == 200:
    print(resp)
else:
    print("Fetch data failed")

# Sample POST Request for model training
# print(" POST Request for model training ")
# api_request = {
#     "model_name": "efficientnetv2-b0",
#     "fine_tune_flag": False
# }
# status, resp = do_post_request(json_dict=api_request, end_point="model_train")
# if status == 200:
#     print(resp)
# else:
#     print("Fetch data failed")