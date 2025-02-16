import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib

# ------------------------
# 1. Load and preprocess the Dataset
# ------------------------

try:
    dataset = pd.read_csv("Iris.csv", encoding='latin1')
except:
    print("can't find data")
    exit()

# check for missing values
missing_values = dataset.isnull().sum()

if(missing_values.any()):
    # Handle missing values 
    inputer = SimpleImputer(strategy='mean')
    dataset_inputer = pd.DataFrame(inputer.fit_transform(dataset), columns=dataset.columns)
    print(dataset_inputer.head())
else:
    print("No missing values")
    dataset_inputer = dataset


# -------------------------------
# 2. Define Target Variables and Features
# ------------------------------
X_features = dataset_inputer[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_target = dataset_inputer['Species']

# -------------------------------
# 3. Train and Compare Models
# ------------------------------

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

#train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
x_pred = model.predict(X_train)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, x_pred)

# save the model
joblib.dump(model, 'Iris_prediction.pkl')

#-------------------------
# 4. Display Results
#-------------------------

print("------------ Iris ---------------\n")

print(f"Test Accuracy: {test_accuracy}")
print(f"Train Accuracy: {train_accuracy}")


# Test using save model
loaded_model = joblib.load('Iris_prediction.pkl')

# create input data
input_data = pd.DataFrame({
    'SepalLengthCm': [5.1],
    'SepalWidthCm': [3.5],
    'PetalLengthCm': [1.4],
    'PetalWidthCm': [0.2]
})

print("\n\n----------PREDICTION BASED ON USER INPUT----------\n")
predicted_result = loaded_model.predict(input_data)
print(f"users input:  {input_data.to_dict( orient='records')}")
print(f"Predicted Result:  {predicted_result}")
