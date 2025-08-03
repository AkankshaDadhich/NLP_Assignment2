import json
import numpy as np

def get_data(file_path):
    X = []
    Y = []
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            chunk_tags = data["chunk_tags"]
            pos_tags = data["pos_tags"]
            pos_tags.insert(0, 0)
            no_of_words = len(chunk_tags)
            x = []
            y = []
            for i in range(no_of_words):
                xi = [0 for _ in range(9)]
                xi[pos_tags[i]] = 1
                xi[pos_tags[i + 1] + 4] = 1
                yi = chunk_tags[i]
                x.append(xi)
                y.append(yi)
            x = np.array(x)
            y = np.array(y)
            X.append(x)
            Y.append(y)
    return X,Y


if __name__ == '__main__':
    X, Y = get_data("Assignment-2\\train.jsonl")
    for x,y in zip(X, Y):
        print(x, y)