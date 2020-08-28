from flask import Flask, request, jsonify
import random
import os
import pickle
from fastapi import FastAPI
import numpy as np
from abnbsfbw import predict1


app = FastAPI(
    title='DS API Airbnb aug-2020',
    description='Under review by Henry Gultom',
    version='0.1',
    docs_url='/',
)

@app.get('/')
def root():
    """hello from Henry, DS17"""
    return {"Hello" : "everyone"}


# Just in case we need user info
@app.get('/user/{username}')
def user(username):
    # return "Hello user"
    object_to_return = {
        "username":username,
        "num_of_requests":random.randint(0, 9),
    }
    return object_to_return # jsonify(object_to_return)


@app.post('/predict')
def predict(accommodates:int = 2, bedrooms: int = 1, bathrooms:float = 1.5,  beds: int=1, guests_included: int=1, minimum_nights: int=1, maximum_nights: int=2):
    """ On this page we will predict the price based on below features"""

    pred = predict1(accommodates, bedrooms, bathrooms, beds,	minimum_nights,	maximum_nights,	guests_included)
    # y_pred = random.randint(20, 300)##temporary number
    res = {
        "status": "success",
        "Accommodates" : accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,    
        "Beds": beds,
        "Guest_included": guests_included,
        "Minimum_nights": minimum_nights,
        "Maximum_nights": maximum_nights,
        "predicted_price": float(pred)
        # "predicted_price": predict(Accommodates, Bathrooms, Bedrooms, Beds, Minimum_Nights , Maximum_Nights, Guests_Included)
    }

    return res


