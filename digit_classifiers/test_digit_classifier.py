import numpy as np
from sklearn import datasets


def test_digit_classifier(classifier, test_start=0, test_count=1000):
    digits = datasets.load_digits()
    correct = 0
    test_end = test_start + test_count
    for img, target in zip(digits.images[test_start:test_end],
                            digits.target[test_start:test_end]):
        v = np.matrix.flatten(img) / 15.
        output = classifier(v)
        answer = list(output).index(max(output))
        if answer == target:
            correct += 1
    return (correct/test_count)

