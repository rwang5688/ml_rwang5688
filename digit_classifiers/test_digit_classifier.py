import numpy as np
from sklearn import datasets


def test_digit_classifier(classifier, test_count=1000):
    digits = datasets.load_digits()
    correct = 0
    for img, target in zip(digits.images[:test_count], digits.target[:test_count]):
        v = np.matrix.flatten(img) / 15.
        output = classifier(v)
        answer = list(output).index(max(output))
        if answer == target:
            correct += 1
    return (correct/test_count)

