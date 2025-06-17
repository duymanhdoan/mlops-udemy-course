import json 
import requests 

url = 'http://66d3-35-192-210-102.ngrok.io/predict' 

request_data = json.dumps({'age':42, 'salary':50000}) 
response = requests.post(url, request_data) 
print (response.text)