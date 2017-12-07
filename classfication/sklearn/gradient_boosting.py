from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

iris_data = load_iris()
X, y = iris_data.data, iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# train model
model = GradientBoostingClassifier(n_estimators=200, max_depth=3)
model.fit(X_train, y_train)

# test model
predictions = model.predict(X_test)

# score on test data (accuracy)
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions, average='weighted')
print('Accuracy: %.4f' % accuracy)
print('Recall: %.4f' % recall)
