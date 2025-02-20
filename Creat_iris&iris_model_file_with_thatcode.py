import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Iris dataset from sklearn
iris = load_iris()

# Create a DataFrame from iris data
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target

# Save the DataFrame as iris.csv
iris_data.to_csv('iris.csv', index=False)
print("iris.csv created successfully!")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model as iris_model.pkl
joblib.dump(model, 'iris_model.pkl')
print("iris_model.pkl created successfully!")
