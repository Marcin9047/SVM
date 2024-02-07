import numpy as np
from dataclasses import dataclass


class Kernel_function:
    def __init__(self, type: str, param1: float, param2: float = None):
        """
        Kernal types: "Polynomial", "Gausian", "Linear"
        """
        self.kernel_type = type
        self.kernel_param1 = param1
        self.kernel_param2 = param2

    def kernel(self, x1, x2):
        param1 = self.kernel_param1
        param2 = self.kernel_param2

        if self.kernel_type == "Polynomial":
            result = np.dot(x1, x2) + param1
            return pow(result, param2)
        elif self.kernel_type == "Gausian":
            distance = np.linalg.norm(x1 - x2)
            return np.exp(-(distance**2) / (2 * param1**2))
        elif self.kernel_type == "Linear":
            return np.dot(x1, x2) + param1


@dataclass
class Model_hyperparams:
    learning_rate: float
    grad_imax: int
    grad_error: float
    alpha_limit: int


class SVM_algorithm:
    def __init__(self, kernel_class: Kernel_function, hyperparams: Model_hyperparams):
        self.learning_rate = hyperparams.learning_rate
        self.imax = hyperparams.grad_imax
        self.grad_error = hyperparams.grad_error
        self.kernel = kernel_class.kernel
        self.alpha_limit = hyperparams.alpha_limit

    def fit(self, Xin, Yin):
        self.x_train = Xin
        self.y_train = Yin
        self.alpha = [0] * len(self.x_train[0])
        self.alpha = self.gradient_descent(self.alpha)

        self.w = 0
        for i in range(len(self.alpha)):
            self.w += self.alpha[i] * self.y_train[i] * self.x_train[i]

        bias_sum = 0
        for i in range(len(self.y_train)):
            bias_new = self.y_train[i] - np.dot(self.w, self.x_train[i])
            bias_sum += bias_new
        self.bias = bias_sum / len(self.y_train)

    def predict(self, Xin):
        decision_function_result = np.dot(self.w, Xin) + self.bias
        prediction = np.sign(decision_function_result)
        true_prediction = np.where(prediction <= -1, -1, 1)
        return true_prediction

    def maximalization_function(self, alpha: list):
        y = self.y_train
        x = self.x_train
        functionOut = 0
        N = len(alpha)
        for i in range(N):
            for j in range(N):
                kernal = self.kernel(x[i].T, x[j])
                functionOut = (alpha[i] * alpha[j] * y[i] * y[j] * kernal) / 2
            functionOut -= alpha[i]
        return functionOut

    def grad_of_maximal(self, alpha):
        x = self.x_train
        y = self.y_train
        N = len(alpha)
        grad = [1] * N
        for Ind in range(N):
            grad[Ind] = 1
            for n in range(N):
                kernal = self.kernel(x[n], x[Ind])
                grad[Ind] -= y[Ind] * y[n] * alpha[n] * kernal
        return grad

    def gradient_descent(self, alphaIn):
        alpha = alphaIn
        beta = self.learning_rate
        t = self.imax
        grad_error = self.grad_error
        for i in range(t):
            grad1 = self.grad_of_maximal(alpha)
            if abs(grad1[0]) < grad_error and i > 100:
                break
            else:
                for ind in range(len(alpha)):
                    if (
                        alpha[ind] + grad1[ind] * beta >= 0
                        and alpha[ind] + grad1[ind] * beta < self.alpha_limit
                    ):
                        alpha[ind] += grad1[ind] * beta
        return alpha
