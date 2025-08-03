import numpy as np
from SP_RNN import SP_RNN
from data_preprocessor import get_data
from tabulate import tabulate

np.random.seed(7)

train_data_path = "Assignment-2\\train.jsonl"
test_data_path = "Assignment-2\\test.jsonl"
weights_file = "Assignment-2\\model_weights.npz"


def train_and_test():

    print("============================================================")
    print("Train and Test")
    chunk_tagger = SP_RNN(no_of_inputs=9)
    chunk_tagger.load_weights(weights_file)

    print("\nBefore Training:")
    X, Y = get_data(train_data_path)
    print("Train Accuracy: ", chunk_tagger.accuracy(X, Y))
    X, Y = get_data(test_data_path)
    print("Test Accuracy: ", chunk_tagger.accuracy(X, Y))

    print("\nTraining:")
    X, Y = get_data(train_data_path)
    chunk_tagger.train(X, Y, no_of_epoch=1, learning_rate=0.01)
    chunk_tagger.save_weights(weights_file)

    print("\nAfter Training:")
    X, Y = get_data(train_data_path)
    print("Train Accuracy: ", chunk_tagger.accuracy(X, Y))
    X, Y = get_data(test_data_path)
    print("Test Accuracy: ", chunk_tagger.accuracy(X, Y))
    print("============================================================")

    print("============================================================")
    print("Test Results:\n")
    X, Y = get_data(test_data_path)
    print("Confusion Matrix:")
    print(chunk_tagger.get_confusion_matrix(X, Y))
    print("Precision:")
    print(chunk_tagger.get_precision(X, Y))
    print("Recall:")
    print(chunk_tagger.get_recall(X, Y))
    print("F1 Score:")
    print(chunk_tagger.get_f1score(X, Y))
    print("Accuracy:")
    print(chunk_tagger.get_accuracy(X, Y))
    print("============================================================")


def k_fold():

    X, Y = get_data(train_data_path)

    n = len(X)
    XX = [ X[:n//5], X[n//5:2*n//5], X[2*n//5:3*n//5], X[3*n//5:4*n//5], X[4*n//5:] ]
    YY = [ Y[:n//5], Y[n//5:2*n//5], Y[2*n//5:3*n//5], Y[3*n//5:4*n//5], Y[4*n//5:] ]
    
    for i in range(5):

        print("============================================================")
        print("Fold : ", i+1)
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for j in range(5):
            if i == j:
                X_test.extend(XX[j])
                Y_test.extend(YY[j])
            else:
                X_train.extend(XX[j])
                Y_train.extend(YY[j])


        chunk_tagger = SP_RNN(no_of_inputs=9)
        chunk_tagger.train(X_train, Y_train, no_of_epoch=1, learning_rate=0.01)
        print("Train Accuracy: ", chunk_tagger.accuracy(X_train, Y_train))
        print("Test Accuracy: ", chunk_tagger.accuracy(X_test, Y_test))
        print("============================================================")


def check_inequalities():

    data = np.load(weights_file)
    input_weights = data["input_weights"]
    feedback_weights = data["feedback_weights"]
    bias_weights = data["bias_weights"]

    THETHA = -bias_weights[0]
    W = feedback_weights[0]
    V_HAT, V_OT, V_NN, V_JJ, V_DT, W_OT, W_NN, W_JJ, W_DT = input_weights

    headers = ["Variable", "Value"]
    table = [
        ["THETHA", THETHA],
        ["W", W],
        ["V_HAT", V_HAT],
        ["V_OT", V_OT],
        ["V_NN", V_NN],
        ["V_JJ", V_JJ],
        ["V_DT", V_DT],
        ["W_OT", W_OT],
        ["W_NN", W_NN],
        ["W_JJ", W_JJ],
        ["W_DT", W_DT]
    ]
    print(tabulate(table, headers=headers, tablefmt="grid"))

    conditions = [
        ("V_HAT+W_DT>THETHA", bool(V_HAT+W_DT>THETHA)),
        ("V_HAT+W_JJ>THETHA", bool(V_HAT+W_JJ>THETHA)),
        ("V_HAT+W_NN>THETHA", bool(V_HAT+W_NN>THETHA)),
        ("V_HAT+W_OT>THETHA", bool(V_HAT+W_OT>THETHA)),
        ("W+V_DT+W_JJ<THETHA", bool(W+V_DT+W_JJ<THETHA)),
        ("W+V_DT+W_NN<THETHA", bool(W+V_DT+W_NN<THETHA)),
        ("V_JJ+W_JJ<THETHA", bool(V_JJ+W_JJ<THETHA)),
        ("V_JJ+W_NN<THETHA", bool(V_JJ+W_NN<THETHA)),
        ("W+V_JJ+W_JJ<THETHA", bool(W+V_JJ+W_JJ<THETHA)),
        ("W+V_JJ+W_NN<THETHA", bool(W+V_JJ+W_NN<THETHA)),
        ("V_NN+W_OT>THETHA", bool(V_NN+W_OT>THETHA)),
        ("W+V_NN+W_OT>THETHA", bool(W+V_NN+W_OT>THETHA)),
        ("W+V_OT+W_DT>THETHA", bool(W+V_OT+W_DT>THETHA)),
        ("W+V_OT+W_JJ>THETHA", bool(W+V_OT+W_JJ>THETHA)),
        ("W+V_OT+W_NN>THETHA", bool(W+V_OT+W_NN>THETHA)),
        ("W+V_OT+W_OT>THETHA", bool(W+V_OT+W_OT>THETHA))
    ]
    headers = ["Condition", "Result"]
    print(tabulate(conditions, headers=headers, tablefmt="grid"))


if __name__ == "__main__":

    train_and_test()
    k_fold()
    check_inequalities()