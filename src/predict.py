import joblib
import pandas as pd 
def predict_yield(avg_temp, avg_rainfall,pesticides):
    model=joblib.load("../models/trained_model.joblib")
    data=pd.DataFrame({
        'Avg_Temp':[avg_temp],
        'Avg_rainfall':[avg_rainfall],
        'Pesticides(tonnes)':[pesticides],

    }) 

    prediction=model.predict(data)
    return prediction[0]