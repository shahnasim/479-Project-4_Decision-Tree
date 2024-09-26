import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('overdrawn.csv')

# Preprocess the data
# Convert DaysDrink into categorical data
conditions = [
    (data['DaysDrink'] < 7),
    (data['DaysDrink'] >= 14),
    (data['DaysDrink'] >= 7) & (data['DaysDrink'] < 14)
]
categories = [0, 2, 1]
data['DaysDrink'] = np.select(conditions, categories)

# Split data into features and target variable
X = data[['Age', 'Sex', 'DaysDrink']]
y = data['Overdrawn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and fit the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Display decision tree
plt.figure(figsize=(10, 5))
plot_tree(clf, filled=True, feature_names=X.columns,
          class_names=['No', 'Yes'], fontsize=5)
plt.show()

# Predict function


def predict_overdrawn(age, sex, days_drink):
    # Preprocess input data
    if days_drink < 7:
        days_drink_cat = 0
    elif days_drink >= 14:
        days_drink_cat = 2
    else:
        days_drink_cat = 1

    # Make prediction
    prediction = clf.predict([[age, sex, days_drink_cat]])
    if prediction[0] == 0:
        return "No"
    else:
        return "Yes"


# Predictions / Queries
predictions = [
    (20, 0, 10),  # 20-year-old male student, drank for 10 days
    (25, 1, 5),   # 25-year-old female student, drank for 5 days
    (19, 0, 20),  # 19-year-old male student, drank for 20 days
    (22, 1, 15),  # 22-year-old female student, drank for 15 days
    (21, 0, 20)   # 21-year-old male student, drank for 20 days
]

for i, (age, sex, days_drink) in enumerate(predictions, 1):
    result = predict_overdrawn(age, sex, days_drink)
    print(
        f"Prediction {i}: Will the student overdraw a checking account? {result}")
