from SVM import SVM_algorithm, Model_hyperparams, Kernel_function
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Creating data frame

df = pd.read_csv("winequality-red.csv")
df1 = df.copy()
df.loc[df["quality"] < 6, "quality"] = -1
df.loc[df["quality"] >= 6, "quality"] = 1
X_values = df.iloc[:, :-1]
Y_values = df.iloc[:, -1]

# Training and testing split
x_train, X_test, y_train, y_test = train_test_split(
    X_values.to_numpy(), Y_values.to_numpy(), test_size=0.6, random_state=43
)
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
X_test_std = sc.transform(X_test)

x = x_train_std

one_counter = 0
for one in y_train:
    if one == 1:
        one_counter += 1

print("\n")
print("Train sample:")
print("1 in sample:", one_counter)
print("0 in sample:", len(y_train) - one_counter)
print("\n")

# SVM parameters
learning_rate = 0.02
lambda_param = 0.3
imax = 1000
limit = 100000
grad_error = 0.0003
kernel = "Gausian"
kernel_param1 = 10.4
kernel_param2 = 4

# Create SVM instance
kernel1 = Kernel_function(kernel, kernel_param1, kernel_param2)
hiperparams = Model_hyperparams(learning_rate, imax, grad_error, limit)
svmtest = SVM_algorithm(kernel1, hiperparams)

# Train the SVM model
svmtest.fit(x_train_std, y_train)

# Test the SVM model with a new data point
prediction = []
x_data = X_test_std
for one in x_data:
    prediction.append(svmtest.predict(one))


one_counter = 0
for one in prediction:
    if one == 1:
        one_counter += 1

print("Test result:")
print("1 predictions:", one_counter)
print("0 prediction:", len(prediction) - one_counter)

one_counter = 0
for one in y_test:
    if one == 1:
        one_counter += 1

print("\n")
print("True values:")
print("1 true value:", one_counter)
print("0 true value:", len(y_test) - one_counter)

# Acuraccy score
print("\n")
print("Accuracy_score:", accuracy_score(y_test, prediction))


svm = SVC(kernel="rbf", random_state=1, C=0.1)
svm.fit(x_train_std, y_train)
y_pred = svm.predict(X_test_std)
print("SKlearn accuracy score: %.3f" % accuracy_score(y_test, y_pred))
