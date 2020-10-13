import numpy as np
from sklearn import datasets


def test_digit_classifier(classifier, test_count=1000):
    digits = datasets.load_digits()
    correct = 0 #<1>
    for img, target in zip(digits.images[:test_count], digits.target[:test_count]): #<2>
        v = np.matrix.flatten(img) / 15. #<3>
        output = classifier(v) #<4>
        answer = list(output).index(max(output)) #<5>
        if answer == target:
            correct += 1 #<6>
    return (correct/test_count) #<7>

