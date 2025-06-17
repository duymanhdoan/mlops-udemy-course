""" 
chú thích ở đây 
"""
from flask import Flask, request 
import numpy as np 
import pickle 
local_classifier = pickle.load(open('classifier.pickle','rb')) 
local_scaler = pickle.load(open('sc.pickle','rb'))


app = Flask(__name__) 

@app.route('/model', methods=['POST']) 

def hello_world():
    request_data = request.get_json(force=True) 
    age = request_data['age'] 
    salary = request_data['salary']
    print(age)
    print(salary)

    prediction = local_classifier.predict(local_scaler.transform(np.array([[age,salary]])))
    return "The prediction is {} model".format(prediction)


if __name__ == "__main__": 
    app.run(port=8005, debug=True)

    
