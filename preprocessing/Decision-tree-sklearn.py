from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import export_text


def print_tree(model):
    tree = export_text(model)
    print(tree)


# Load the iris dataset
iris = pd.read_csv('../data/iris.csv')
iris.drop(iris[iris['species'] == 'virginica'].index,axis=0, inplace=True)
X = iris.drop('species',axis=1)
print(type(X))
y = iris['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=10)

# Train the classifier on the training set
clf.fit(X_train, y_train)

# Predict on the testing set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print_tree(clf)
