import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("insurance.csv")

# Encode Categorical Columns
le = LabelEncoder()
for col in ["sex", "smoker", "region"]:
    df[col] = le.fit_transform(df[col])

# Features and Target
x = df.drop("charges", axis=1)
y = df["charges"]

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluation
print("MSE :", mean_squared_error(y_test, y_pred))
print("R² Score :", r2_score(y_test, y_pred))

# Coefficients
print("\nCoefficients:")
for feature, coef in zip(x.columns, model.coef_):
    print(feature, ":", coef)

# Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue")

# Red Reference Line
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red",
    linewidth=2
)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted")
plt.show()

# Predict New Customer
new_customer = pd.DataFrame({
    "age": [30],
    "sex": [1],
    "bmi": [28.5],
    "children": [2],
    "smoker": [0],
    "region": [2]
})

prediction = model.predict(new_customer)

print("\nPredicted Insurance Charge:", prediction[0])