from DoSomething import DoSomething
import json
from datetime import datetime
import time

class Subscriber(DoSomething):
    def notify(self, topic, msg):
        
        msg_json = json.loads(msg)
        timestamp = int(msg_json['bt'])
        time_str = str(datetime.fromtimestamp(timestamp))
        n = msg_json['e'][0]['n']
        u = msg_json['e'][0]['u']
        y_true = msg_json['e'][0]['v']
        y_pred = msg_json['e'][1]['v']

        print ("("+str(time_str)+") " + n +": "+ "Predicted={:.3f}".format(y_pred)+u+" Actual={}".format(y_true)+u)

        
        


if __name__ == '__main__':
    subscriver = Subscriber('subscriber 1')
    subscriver.run() 
    subscriver.myMqttClient.mySubscribe('/Group17/alert')
    while(True):
        time.sleep(1)