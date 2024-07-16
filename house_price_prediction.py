import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Display the first few rows of the train dataset
print(train_df.head())

# Select features and target from the train dataset
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X = train_df[features]
y = train_df[target]

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plotting the results
plt.figure(figsize=(20, 6))

# Scatter plot for the actual vs predicted prices
plt.subplot(1, 3, 1)
plt.scatter(y_val, y_pred, color='blue', label='Predicted Prices')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--', label='Ideal Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.legend()

# Scatter plot for square footage vs predicted prices
plt.subplot(1, 3, 2)
plt.scatter(X_val['GrLivArea'], y_val, color='blue', label='Actual Prices')
plt.scatter(X_val['GrLivArea'], y_pred, color='red', alpha=0.5, label='Predicted Prices')
plt.xlabel('Square Footage')
plt.ylabel('Prices')
plt.title('Square Footage vs Prices')
plt.legend()

# Scatter plot for bedrooms and bathrooms vs predicted prices
plt.subplot(1, 3, 3)
plt.scatter(X_val['BedroomAbvGr'] + X_val['FullBath'], y_val, color='blue', label='Actual Prices')
plt.scatter(X_val['BedroomAbvGr'] + X_val['FullBath'], y_pred, color='red', alpha=0.5, label='Predicted Prices')
plt.xlabel('Number of Bedrooms and Bathrooms')
plt.ylabel('Prices')
plt.title('Bedrooms and Bathrooms vs Prices')
plt.legend()

plt.tight_layout()
plt.show()

# Display the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Make predictions on the test set
test_predictions = model.predict(test_df[features])

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_predictions
})

# Save the submission to a CSV file
submission_df.to_csv('sample_submission.csv', index=False)
