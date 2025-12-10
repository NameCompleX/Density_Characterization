import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# Read data from a tab-separated file
file_path = "Your file path here"
data = pd.read_csv(file_path, sep="\t", header=None, names=["x", "y"])

# Extract x and y values
x = data["x"].values.reshape(-1, 1)
y = data["y"].values

# Set up and fit the linear regression model
model = LinearRegression()
model.fit(x, y)
slope = model.coef_[0]
intercept = model.intercept_

# Print the results
print(f"Equation: y = {slope:.4f}x + {intercept:.4f}")
score = model.score(x, y)
print("Score:", score)

plt.scatter(x, y, color="blue", label="Real Density")
plt.plot(x, model.characterize(x), color="red", label="Regression line")
plt.xlabel("Gray Scale")
plt.ylabel("Density")
plt.legend()
plt.title("Linear Regression")
plt.show()


