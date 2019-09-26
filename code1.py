import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target
X_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2)


# Using SVM
# kernel takes it up to an additional dimension; kernel = poly takes too much time; C -> margins
clfsvm = svm.SVC(kernel="linear", C=2)
clfsvm.fit(X_train, y_train)
y_predsvm = clfsvm.predict(x_test)
accuracysvm = metrics.accuracy_score(y_test, y_predsvm)
print("The accuracy with SVM classifier is: ")
print(accuracysvm)

# Using KNN
clfknn = KNeighborsClassifier(n_neighbors=5)
clfknn.fit(X_train, y_train)
y_predknn = clfknn.predict(x_test)
accuracyknn = metrics.accuracy_score(y_test, y_predknn)
print("The accuracy with KNN classifier is: ")
print(accuracyknn)
