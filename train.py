import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ------------------------
# 1. Load and preprocess the Dataset
# ------------------------

try:
    dataset = pd.read_csv("HousingData.csv", encoding='latin1')
except:
    print("cant find data")
    exit()


# -Check for missing values
missing_values = dataset.isnull().sum()

if(missing_values.any()):
    # -Handle missing values 
    inputer = SimpleImputer(strategy='mean')
    dataset_inputer = pd.DataFrame(inputer.fit_transform(dataset), columns=dataset.columns)
else :
    print("No missing values")
    


# -------------------------------
# 2. Define Target Variables and Features
# ------------------------------

#  features and their descriptions 
'''
    'CRIM': 'per capita crime rate by town',
    'ZN': 'proportion of residential land zoned for lots over 25,000 sq.ft.',
    'INDUS': 'proportion of non-retail business acres per town',
    'CHAS': 'Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)',
    'NOX': 'nitric oxides concentration (parts per 10 million)',
    'RM': 'average number of rooms per dwelling',
    'AGE': 'proportion of owner-occupied units built prior to 1940',
    'DIS': 'weighted distances to five Boston employment centres',
    'RAD': 'index of accessibility to radial highways',
    'TAX': 'full-value property-tax rate per $10,000',
    'PTRATIO': 'pupil-teacher ratio by town',
    'B': '1000(Bk - 0.63)^2 where Bk is the proportion
    "LSTAT": "percentage of lower status of the population",
    'MEDV': 'Median value of owner-occupied homes in $1000s'
'''
# selecting features and target variables
X_features = dataset_inputer[[ 'RM', 'LSTAT']]
y_target = dataset_inputer['MEDV']

# -------------------------------
# 3. Train and Compare Models
# ------------------------------

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=42)

# Train the Model
model = LinearRegression()

# hyperparameters grid for Grid Search
param_grid = {
    'fit_intercept': [True, False],
}

grid_search = GridSearchCV(model, param_grid, cv=5)

#perform Grid Search with cross-validation
grid_search.fit(X_train, y_train)



best_model = grid_search.best_estimator_
# Predict on test set
test_prediction = best_model.predict(X_test)

# Predict on training set
yprediction = best_model.predict(X_train)

# Evaluate the Model on test set
test_score = best_model.score(X_test, y_test)
test_mse = mean_squared_error(y_test, test_prediction)


print("----------------BOSTON HOUSE------------------ \n")

print(f"Test Set - Model mean Accuracy: {test_score}")
print(f"Test Set - Mean Squared Error: {test_mse}")

print("---------------------------------------------")
# Evaluate the Model on training set
train_score = best_model.score(X_train, y_train)
train_mse = mean_squared_error(y_train, yprediction)

print(f"Training Set - Model mean Accuracy: {train_score}")
print(f"Training Set - Mean Squared Error: {train_mse}")

# save the model
joblib.dump(best_model, 'BostonHouse_prediction.pkl')


# Test using save model
loaded_model = joblib.load('BostonHouse_prediction.pkl')

#create input data
input_data = pd.DataFrame({
    'RM': [6.575],
    'LSTAT': [4.98]
})

print("\n\n----------PREDICTION BASED ON USER INPUT----------\n")

predicted_result = loaded_model.predict(input_data)
print(f"users input-  {input_data.to_dict(orient='records')}")
print(f"Predicted Result-  {predicted_result}")
print(f"interpretation-  {predicted_result[0]:.2f} Median value of owner-occupied homes in $1000s \n\n\n")



## Plot the results

# Plot Actual vs Predicted Value
plt.figure(figsize=(10, 5))
plt.scatter(y_test, test_prediction, color='blue', label='Actual vs Predicted', alpha=0.5)
plt.plot([y_target.min(), y_target.max()], [y_target.min(), y_target.max()], color='red', lw=2, label='Perfect Prediction')
plt.title('Actual vs Predicted Value')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.savefig('actual_vs_predicted.png')

# Plot Residuals
residuals = y_test - test_prediction
plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.savefig('residuals_distribution.png')






