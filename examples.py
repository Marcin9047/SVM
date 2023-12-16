from SVM import SVM_algorithm, Model_Hiperparams, Kernel_function
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import shap
from sklearn.metrics import precision_recall_curve, average_precision_score

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
hiperparams = Model_Hiperparams(learning_rate, imax, grad_error, limit)
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

print("Test result::")
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


# fig, ax = plt.subplots(figsize=(12, 8))
# sns.heatmap(df1.corr(), annot=True, ax=ax)
# plt.title("Wykres korelacji")
# plt.show()


# target = "quality"
# features = [col for col in df1.columns.to_list() if col != target]

# # dla każdego feature rysujemy boxplot
# fig, axis = plt.subplots(2, df.shape[1] // 2, figsize=[20, 10])
# idx = 0
# axis = axis.flatten()
# for feature in features:
#     sns.boxplot(y=feature, data=df[[feature]], ax=axis[idx])
#     idx += 1

# plt.tight_layout()
# plt.show()

# fig, axis = plt.subplots(df1.shape[1] // 2, 2, figsize=[20, 40])
# idx = 0
# axis = axis.flatten()
# for feature in features:
#     df1.plot(y=target, x=feature, kind="scatter", ax=axis[idx])
#     idx += 1

# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(10, 10))
# sns.scatterplot(x="pH", y="alcohol", data=X_values[["pH", "alcohol"]])
# # tmp_x = np.array(
# x1 = X_values.loc[X_values["pH"] == X_values["pH"].min()].to_numpy()
# x2 = X_values.loc[X_values["alcohol"] == X_values["alcohol"].max()].to_numpy()
# tmp2_x = np.array([x2, x2])
# print(x1)
# tmp_y1 = svmtest.predict(x1)
# tmp_y2 = svmtest.predict(x2)
# plt.plot(tmp_x, tmp_y, color="r")
# plt.show()
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)

# # Wygeneruj wykres ROC
# plt.figure()
# plt.plot(
#     fpr,
#     tpr,
#     color="darkorange",
#     lw=2,
#     label="ROC curve (area = {:.2f})".format(roc_auc),
# )
# plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic")
# plt.legend(loc="lower right")
# plt.show()


# precision, recall, _ = precision_recall_curve(y_test, y_pred)


# plt.figure()
# plt.step(recall, precision, color="b", where="post")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall curve")
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.show()
# def accuracy_score_for_param():
#     accuracy_vec = []
#     learning_rate = 0.02
#     imax = 1000
#     limit = 100000
#     grad_error = 0.0003
#     kernel = "Gausian"
#     kernal_param1 = 10
#     xexis = []
#     for i in range(10):
#         learning_rate = 0.01 + 0.01 * i
#         # Create SVM instance
#         hiperparams = Model_Hiperparams(
#             learning_rate, imax, grad_error, limit, kernel, kernal_param1, kernal_param2
#         )
#         svmtest = SVM_algorithm(hiperparams)
#         svmtest.fit(x_train_std, y_train)

#         # Test the SVM model with a new data point
#         prediction = []
#         x_data = X_test_std
#         for one in x_data:
#             prediction.append(svmtest.predict(one))
#         accuracy_vec.append(accuracy_score(y_test, prediction))
#         xexis.append(learning_rate)

#     plt.plot(xexis, accuracy_vec, color="r")


# accuracy_score_for_param()
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()

# # residuals = y_test - y_pred
# # sns.residplot(y_pred, residuals, lowess=True, line_kws={"color": "red"})
# plt.title("Gausian accuracy based on gradient learning rate")
# plt.xlabel("learning rate")
# plt.ylabel("Accuracy score")
# plt.show()

# explainer = shap.Explainer(svmtest)
# shap_values = explainer.shap_values(X_test_std)

# shap.summary_plot(shap_values, X_test_std)
# precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# # Oblicz pole powierzchni pod krzywą Precision-Recall (AUC-PR)
# average_precision = average_precision_score(y_test, y_pred)

# # Wygeneruj wykres Precision-Recall Curve
# plt.plot(
#     recall,
#     precision,
#     color="b",
#     lw=2,
#     label="Precision-Recall curve (AUC = {:.2f})".format(average_precision),
# )
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.legend(loc="lower left")
# plt.show()

# Sample data (replace with your actual data)

# Compute ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# # List of threshold indices you want to highlight
# highlight_indices = [1, 3, 5, 7]

# Plot ROC curve
# plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")

# # Plot highlighted points on the ROC curve
# plt.scatter(
#     fpr[highlight_indices],
#     tpr[highlight_indices],
#     c="red",
#     marker="o",
#     label="Highlighted Points",
# )

# # Plot diagonal line
# plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

# # Annotate highlighted points with their corresponding thresholds
# for i in highlight_indices:
#     plt.annotate(
#         f"Threshold={thresholds[i]:.2f}",
#         (fpr[i], tpr[i]),
#         textcoords="offset points",
#         xytext=(0, 10),
#         ha="center",
#         fontsize=8,
#     )

# plt.xlabel("False Positive Rate (FPR)")
# plt.ylabel("True Positive Rate (TPR)")
# plt.title("Receiver Operating Characteristic (ROC) Curve")
# plt.legend(loc="lower right")
# plt.show()
