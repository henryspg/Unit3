import random
import os
import pickle
from fastapi import FastAPI
import numpy as np
from abnbsfbw import predict1


app = FastAPI(
    title='Data Science API project: Airbnb SF',
    description='By: Henry Gultom, \nLambda School student',
    version='0.1',
    docs_url='/',
)


@app.get('/')
def root():
    """To get the price prediction, please scroll down and click the green box: **POST /predict** """
    return {"Hello" : "everyone"}


# Just in case we need user info
# @app.get('/user/{username}')
# def user(username):
#     # return "Hello user"
#     object_to_return = {
#         "username":username,
#         "num_of_requests":random.randint(0, 9),
#     }
#     return object_to_return


@app.post('/predict')
def predict(accommodates:int = 1, bedrooms: int = 1, bathrooms:float = 1.0,  beds: int=1, guests_included: int=1, minimum_nights: int=1, maximum_nights: int=2):
    """ 
    1. click: **Try it out**  and change the value under the **Description**, and click: **Execute**.\n
    2. It predicts Airbnb rent price using XGBOOST as Machine Learning model.\n
    3. See the result below at: **Responses**  >>  **Details**  >>  **Response body**
    """ 

    pred = predict1(accommodates, bedrooms, bathrooms, beds,	minimum_nights,	maximum_nights,	guests_included)
    res = {
        "Accommodates"      : accommodates,
        "bedrooms"          : bedrooms,
        "bathrooms"         : bathrooms,    
        "Beds"              : beds,
        "Guest_included"    : guests_included,
        "Minimum_nights"    : minimum_nights,
        "Maximum_nights"    : maximum_nights,
        "predicted_price"   : round(float(pred),1)
        }

    return res


