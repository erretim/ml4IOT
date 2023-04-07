import base64 
import requests
import json
import time


def get_base64_encoded_model(model_path):
    with open(model_path, "rb") as model_file:
        return base64.urlsafe_b64encode(model_file.read()).decode()

if __name__ == '__main__':
    
    print("========= 1.1 ===============")
    model_path = './mlp.tflite'
    model_b64str = get_base64_encoded_model(model_path)

    name = 'mlp.tflite'

    body = {
        'name': name, 
        'model': model_b64str
    }
    body = json.dumps(body)

    url = f"http://localhost:8080/add"

    r = requests.put(url,data=body)
    
    if r.status_code == 200:
        print("Code:", r.status_code)
    else:	
        print('Error:', r.status_code)

    

    print("========= 1.2 ===============")
    model_path = './cnn.tflite'
    model_b64str = get_base64_encoded_model(model_path)

    name = 'cnn.tflite'

    body = {
        'name': name, 
        'model': model_b64str
    }
    body = json.dumps(body)

    url = f"http://localhost:8080/add"

    r = requests.put(url,data=body)
    
    if r.status_code == 200:
        print("Code:", r.status_code)
    else:	
        print('Error:', r.status_code)
        exit()

    print("========= 2 ===============")

    url = f"http://localhost:8080/list"

    r = requests.get(url)
    if r.status_code == 200:
        print("Code:", r.status_code)
        body = r.json()

        if len(body['models']) != 2:
            print('Error: wrong number of models')
            exit()

        for model in body['models']:
            print(model)
        

    else:	
        print('Error:', r.status_code)
        exit()
    
    print("========= 3 ===============")

    url = "http://localhost:8080/predict?model=cnn.tflite&tthres=0.1&hthres=0.2"

    r = requests.get(url)
    if r.status_code == 200:
        print("Code:", r.status_code, )
    else:	
        print('Error:', r.status_code)
        exit()

    while True:
        time.sleep(1)