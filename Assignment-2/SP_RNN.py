import numpy as np
from tqdm import tqdm


# Single Perceptron Recurrent Neural Network
class SP_RNN:

    def __init__(self, no_of_inputs, no_of_feedback=1, no_of_bias=1):
        self.no_of_inputs = no_of_inputs
        self.no_of_feedback = no_of_feedback
        self.no_of_bias = no_of_bias
        self.input_weights = 2 * np.random.rand(self.no_of_inputs) - 1
        self.feedback_weights = 2 * np.random.rand(self.no_of_feedback) - 1
        self.bias_weights = 2 * np.random.rand(self.no_of_bias) - 1
        self.bias_input = np.ones(self.no_of_bias)

    def predict_one(self, x):
        feedback_input = np.zeros(self.no_of_feedback)
        y = []
        for xi in x:
            xi = np.array(xi)
            yi = (
                np.dot(xi, self.input_weights)
                + np.dot(feedback_input, self.feedback_weights)
                + np.dot(self.bias_input, self.bias_weights)
            )
            yi = self.sigmoid(yi)
            feedback_input = np.array([yi])
            y.append(yi)
        return y

    def train(self, X, Y, no_of_epoch, learning_rate):
        # print(self.cross_entropy_loss(X, Y))
        for _ in tqdm(range(no_of_epoch)):
            for x, y in zip(X, Y):
                total_change_in_bias_weights = np.zeros(self.no_of_bias)
                total_change_in_input_weights = np.zeros(self.no_of_inputs)
                total_change_in_feedback_weights = np.zeros(self.no_of_feedback)
                y_pred = self.predict_one(x)
                previous_output = np.array([0])
                for xi, yi_a, yi_p in zip(x, y, y_pred):
                    delta_j = (yi_a - yi_p) * yi_p * (1 - yi_p)
                    total_change_in_input_weights += learning_rate * delta_j * xi
                    total_change_in_feedback_weights += (
                        learning_rate * delta_j * previous_output
                    )
                    total_change_in_bias_weights += (
                        learning_rate * delta_j * self.bias_input
                    )
                    previous_output = np.array([yi_p])
                self.input_weights += total_change_in_input_weights
                self.feedback_weights += total_change_in_feedback_weights
                self.bias_weights += total_change_in_bias_weights
        # print(self.cross_entropy_loss(X, Y))

    def cross_entropy_loss(self, X, Y):
        loss = 0
        total = 0
        for x, y in zip(X, Y):
            y_pred = self.predict_one(x)
            for yi_a, yi_p in zip(y, y_pred):
                loss += -(yi_a * np.log(yi_p) + (1 - yi_a) * np.log(1 - yi_p))
                total += 1
        loss /= total
        return loss

    def classify_class(self, y):
        y = [1 if t >= 0.5 else 0 for t in y]
        return y
    
    def predict_output(self, x):
        y = self.predict_one(x)
        y = self.classify_class(y)
        y = np.array(y)
        return y

    def accuracy(self, X, Y):
        correct = 0
        total = 0
        for x, y in zip(X, Y):
            y_pred = self.predict_one(x)
            y_pred = self.classify_class(y_pred)
            for yi_a, yi_p in zip(y, y_pred):
                total += 1
                if yi_a == yi_p:
                    correct += 1

        return correct / total

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def get_confusion_matrix(self, X, Y):
        conf_matrix = [[0, 0], [0, 0]]

        true_labels = Y
        predicted_labels = [self.predict_output(x) for x in X]
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            for true_label_i, predicted_label_i in zip(true_label, predicted_label):
                conf_matrix[true_label_i][predicted_label_i] += 1

        confusion_matrix = {
            'true_positive'     : conf_matrix[1][1],
            'false_positive'    : conf_matrix[0][1],
            'false_negative'    : conf_matrix[1][0],
            'true_negative'     : conf_matrix[0][0]
        }
        return confusion_matrix

    def get_accuracy(self, X, Y):
        confusion_matrix = self.get_confusion_matrix(X, Y)
        true_positive = confusion_matrix['true_positive']
        true_negative = confusion_matrix['true_negative']
        false_positive = confusion_matrix['false_positive']
        false_negative = confusion_matrix['false_negative']
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        return accuracy
    
    def get_precision(self, X, Y):
        confusion_matrix = self.get_confusion_matrix(X, Y)
        true_positive = confusion_matrix['true_positive']
        false_positive = confusion_matrix['false_positive']
        precision = true_positive / (true_positive + false_positive)
        return precision
    
    def get_recall(self, X, Y):
        confusion_matrix = self.get_confusion_matrix(X, Y)
        true_positive = confusion_matrix['true_positive']
        false_negative = confusion_matrix['false_negative']
        recall = true_positive / (true_positive + false_negative)
        return recall
    
    def get_f1score(self, X, Y):
        precision = self.get_precision(X, Y)
        recall = self.get_recall(X, Y)
        f1score = 2 * precision * recall / (precision + recall)
        return f1score
    

    def save_weights(self, file_path):
        np.savez(
            file_path,
            input_weights=self.input_weights,
            feedback_weights=self.feedback_weights,
            bias_weights=self.bias_weights,
        )

    def load_weights(self, file_path):
        data = np.load(file_path)
        self.input_weights = data["input_weights"]
        self.feedback_weights = data["feedback_weights"]
        self.bias_weights = data["bias_weights"]

