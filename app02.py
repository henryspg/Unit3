import random
import os
from fastapi import FastAPI
import numpy as np
from abnbsfbw import predict1


app = FastAPI(
    title='Data Science API project: Airbnb SF',
    description='By: Henry Gultom, Lambda School student',
    version='0.1',
    docs_url='/',
)


@app.get('/')
def root():
    """To get the price prediction, please scroll down and click the green box: **POST /predict** """
    return {"Hello" : "everyone"}


@app.post('/predict')
def predict(accommodates:int = 1, bedrooms: int = 1, bathrooms:float = 1.0,  beds: int=1, guests_included: int=1, minimum_nights: int=1, maximum_nights: int=2):
    """ 
    1. click: **Try it out**  and change the value under the **Description**, and click: **Execute**.\n
    2. It predicts Airbnb rent price using XGBOOST as Machine Learning model.\n
    3. Scroll down to see the result at: **Responses**  >>  **Details**  >>  **Response body**
    """ 

    pred = predict1(accommodates, bedrooms, bathrooms, beds,	minimum_nights,	maximum_nights,	guests_included)
    # pred = random.randint(1,200)
    
    res = {
            "predicted_price"   : round(float(pred),1)
        }

    return res
