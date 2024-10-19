from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Machine Learning Model", description="Serve ML Model with FastAPI")

# Load the model once when the application starts
model = joblib.load('iris_model.pkl')

# Define the mapping of class indices to class names
class_names = ["setosa", "versicolor", "virginica"]

# Prediction function
def make_prediction(sepal_length, sepal_width, petal_length, petal_width):
    data_in = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction_index = model.predict(data_in)[0]  # Get the predicted class index
    probability = model.predict_proba(data_in).max()  # Get the maximum probability
    return class_names[prediction_index], float(probability)  # Map index to class name

# Schema for the input data
class IrisModel(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# POST endpoint for prediction
@app.post("/predict")
def predict(iris: IrisModel):
    data = iris.dict()
    pred, prob = make_prediction(data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width'])
    return {"prediction": pred, "probability": prob}

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Machine Learning Model API!"}
