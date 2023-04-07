import cherrypy 
import json 
import base64
import os
import tensorflow as tf
import numpy as np
from DoSomething import DoSomething 
from datetime import datetime 
import time

from board import D4 
import adafruit_dht 


class AddService(object):
    exposed=True
    def GET(self, *path, **query):
        pass
    def POST(self, *path, **query):
        pass
    
    def PUT(self, *path, **query):
        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'WRONG PATH ERROR')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'WRONG QUERY ERROR')

        body_str = cherrypy.request.body.read()
        body = json.loads(body_str)
        if body is None:
            raise cherrypy.HTTPError(400, 'MISSING BODY ERROR - BODY')    
        
        model_b64str = body['model']
        if model_b64str is None:
            raise cherrypy.HTTPError(400, 'MISSING BODY ERROR - MODEL')
        name = body['name']
        if name is None:
            raise cherrypy.HTTPError(400, 'MISSING BODY ERROR - NAME')
        model_b64 = model_b64str.encode()
        model = base64.urlsafe_b64decode(model_b64)
        models_path = './models/'
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        with open(f"{models_path}/{name}", "wb") as outfile:
            outfile.write(model)
    
    def DELETE(self, *path, **query):
        pass

class ListService(object):
    exposed=True
    def GET(self, *path, **query):
        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'WRONG PATH ERROR')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'WRONG QUERY ERROR')

        models_path = './models/'
        if not os.path.exists(models_path):
            raise cherrypy.HTTPError(400, 'WRONG DIRECTORY ERROR')

        models = os.listdir(models_path)

        body = {
            'models' : models,
        }
        body_json = json.dumps(body)
        return body_json

    def POST(self, *path, **query):
        pass
    def PUT(self, *path, **query):
        pass
    def DELETE(self, *path, **query):
        pass

class PredictService(object):
    
    exposed=True

    def __init__(self):
        self.i = 0
        self.dht_device = adafruit_dht.DHT11(D4)
        self.publisher = DoSomething('publisher 1')
        self.publisher.run()


    def GET(self, *path, **query):
        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')
        if len(query) != 3:
            raise cherrypy.HTTPError(400, 'Wrong query')
        
        model_name = str(query.get('model'))
        if model_name is None:
            raise cherrypy.HTTPError(400, 'WRONG QUERY - model')
            
        tthres = float(query.get('tthres'))
        if tthres is None:
            raise cherrypy.HTTPError(400, 'WRONG QUERY - tthres')
            
        hthres = float(query.get('hthres'))
        if hthres is None:
            raise cherrypy.HTTPError(400, 'WRONG QUERY - hthres')


        model_path = f"./models/{model_name}"
        if not os.path.exists(model_path):
            raise cherrypy.HTTPError(400, 'WRONG QUERY - model')
            
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        

        # values obtained from the same dataset on which the models were trained.
        mean = np.array([ 9.107597, 75.904076], dtype=np.float32)
        std = np.array([ 8.654227, 16.557089], dtype=np.float32)

        x = np.zeros([1, 6, 2], dtype=np.float32)
        y_true = np.zeros(2, dtype=np.float32)


        while(True):
                
            temperature = self.dht_device.temperature 
            humidity = self.dht_device.humidity
            
            if self.i <= 5:
                x[0, self.i , 0] = float(temperature)
                x[0, self.i, 1] = float(humidity)
                self.i += 1
                    
            else:
                y_true[0] = np.float32(temperature)
                y_true[1] = np.float32(humidity)

                temp_x = (x - mean) / std
                interpreter.set_tensor(input_details[0]['index'], temp_x)
                interpreter.invoke()
                y_pred = interpreter.get_tensor(output_details[0]['index'])
                temp_err = abs(y_true[0]-y_pred[0,0])
                hum_error = abs(y_true[1]-y_pred[0,1])
                    
                if temp_err > tthres:
                    timestamp = int(datetime.now().timestamp())
                    body = {
                        "bn":"raspberrypi.local",
                        "bt": timestamp, 
                        "e": [
                            {"n":"Temperature", "u":"C", "t":0, "v":float(y_true[0])},
                            {"n":"Temperature", "u":"C", "t":0, "v":float(y_pred[0,0])},
                        ]
                    }
                    body = json.dumps(body)
                    self.publisher.myMqttClient.myPublish('/Group17/alert', body)
                        

                if hum_error > hthres:
                    timestamp = int(datetime.now().timestamp())
                    body = {
                        "bn":"raspberrypi.local",
                        "bt": timestamp, 
                        "e": [
                            {"n":"Humidity", "u":"%", "t":0, "v":float(y_true[1])},
                            {"n":"Humidity", "u":"%", "t":0, "v":float(y_pred[0,1])},
                        ]
                    }
                    body = json.dumps(body)
                    self.publisher.myMqttClient.myPublish('/Group17/alert', body)
                
                x[0, :, 0] = np.roll(x[0, :, 0], -1)
                x[0, :, 1] = np.roll(x[0, :, 1], -1)
                
                x[0, -1, 0] = float(temperature)
                x[0, -1, 1] = float(humidity)
                

            time.sleep(1)
  
    def POST(self, *path, **query):
        pass
    def PUT(self, *path, **query):
        pass
    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':

    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    
    cherrypy.tree.mount(AddService(), '/add', conf) 
    cherrypy.tree.mount(ListService(), '/list', conf) 
    cherrypy.tree.mount(PredictService(), '/predict', conf) 
    
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port':8080})
    cherrypy.engine.start() 
    cherrypy.engine.block()