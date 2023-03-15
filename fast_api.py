import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from typing import Dict
import numpy as np
import pickle
# import json
import pandas as pd

app=FastAPI()

origins=['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post("/predict")
def predict(data: Dict):
    # Convert the input data to a pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index').T
    print(df)
    # loading the saved model
    # model=pickle.load(open('1d_conv.pkl','rb'))
    model = keras.models.load_model('lstm.hdf5', compile=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Make a prediction using the trained model
    inp = np.asarray(df)
    input_data = inp.reshape(-1,1, 28)
    y_pred = model.predict(input_data)
    print(y_pred)
    # Return the prediction as a dictionary
    return f'{y_pred[0][0]}'

if __name__=='__main__':
    uvicorn.run(app)