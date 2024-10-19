import argparse
import joblib

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-sepal_length", help="sepal_length", type=float)
parser.add_argument("-sepal_width", help="sepal_width", type=float)
parser.add_argument("-petal_length", help="petal_length", type=float)
parser.add_argument("-petal_width", help="petal_width", type=float)

args = parser.parse_args()

# Prediction function
def predict():
    model = joblib.load('iris_model.pkl')
    data_in = [[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]]
    prediction = model.predict(data_in)
    probability = model.predict_proba(data_in).max()
    print("Prediction:", prediction[0], "Probability:", probability)

# Call the predict function
if __name__ == "__main__":
    predict()
