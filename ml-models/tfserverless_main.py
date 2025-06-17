import requests
import pickle
from google.cloud import storage
import tensorflow as tf


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
    bucket = storage_client.get_bucket('fx-tf-models')

    blob_weights1 = bucket.blob('text_classifier_model/1/variables/variables.index')
    blob_weights2 = bucket.blob('text_classifier_model/1/variables/variables.data-00000-of-00001')

    print("loaded models from bucket")

    blob_tfidf = bucket.blob('tfidfmodel.pickle')
    print("loaded tfidf model")

    blob_weights1.download_to_filename('/tmp/variables.index')
    blob_weights2.download_to_filename('/tmp/variables.data-00000-of-00001')

    print("downloaded model weights  ")

    serverless_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    serverless_model.load_weights('/tmp/variables')

    print("Loaded weights  ")


    blob_tfidf.download_to_filename('/tmp/tfidfmodel.pickle')
    print("downloaded tfidf")
 
    serverless_tfidf = pickle.load(open('/tmp/tfidfmodel.pickle','rb'))
    
    text = request_json['setence']
    print("printing the sentence")
    print(text)
    text_list=[]
    text_list.append(text)
    print(text_list)
    numeric_text = serverless_tfidf.transform(text_list).toarray()
    output = serverless_model.predict(numeric_text)[:,1]
    print("Printing prediction")
    print(output)
    sentiment="unknown"
    if output[0] > 0.5 :
      print("positive prediction")
      sentiment="postive"
    else:
      print("negative prediction")
      sentiment="negative"
    print("Printing sentiment")     
    print(sentiment)
    return "The sentiment is {}".format(sentiment)




