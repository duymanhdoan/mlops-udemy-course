import requests
import pickle
from google.cloud import storage

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F



def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('fx-pytorch-demo')
    print("created bucket instance")
    
    blob_dictionary = bucket.blob('text_classifier_pytorch')
    blob_tfidf = bucket.blob('tfidfmodel.pickle')
    blob_dictionary.download_to_filename('/tmp/text_classifier_pytorch')
    blob_tfidf.download_to_filename('/tmp/tfidfmodel.pickle')
  
    serverless_tfidf = pickle.load(open('/tmp/tfidfmodel.pickle','rb'))
    
    print("loaded scaler from bucket")

    input_size=467
    output_size=2
    hidden_size=500

    class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.fc1 = torch.nn.Linear(input_size, hidden_size)
           self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
           self.fc3 = torch.nn.Linear(hidden_size, output_size)


       def forward(self, X):
           X = torch.relu((self.fc1(X)))
           X = torch.relu((self.fc2(X)))
           X = self.fc3(X)

           return F.log_softmax(X,dim=1)

    model = Net()
    
    model.load_state_dict(torch.load('/tmp/text_classifier_pytorch'))

    print("Loaded PyTorch Dictionary  ")

    
    text = request_json['sentence']
    print("printing the sentence")
    print(text)
    text_list=[]
    text_list.append(text)
    print(text_list)
    numeric_text = serverless_tfidf.transform(text_list).toarray()
    output = model(torch.from_numpy(numeric_text).float())
    print("Printing prediction")
    print("Printing predictions")
    print(output[:,0][0])
    print(output[:,1][0])    
    sentiment="unknown"
    
    sentiment="unknown"
    if torch.gt(output[:,0][0],output[:,1][0]):
      print("negative prediction")
      sentiment="negative from pytorch"
    else:
      print("positive prediction")
      sentiment="positive from pytorch"
    print("Printing prediction")     
    print(sentiment)
    
    return "The sentiment is {}".format(sentiment)




