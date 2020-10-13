import numpy as np
from sklearn import datasets
from digit_classifiers import random_digit_classifier
from digit_classifiers import calculate_average_digits, average_digit_classifier
from test_digit_classifier import test_digit_classifier


def main():
    digits = datasets.load_digits()

    # get image0, which happens to be digit '0'
    image0 = digits.images[0]
    target0 = digits.target[0]
    print('image0:')
    print(image0)
    print(f'image0 (target0) should be: {target0}')

    flattened_image0 = np.matrix.flatten(image0)
    print('falttened image0:')
    print(flattened_image0)

    scaled_and_flattened_image0 = flattened_image0 / 15
    print('scaled_and_flattened_image0:')
    print(scaled_and_flattened_image0)

    # use random digit classifier on image0
    print('===')
    result0 = random_digit_classifier(scaled_and_flattened_image0)
    print('random classifier result:')
    print(result0)
    guess0 = list(result0).index(max(result0))
    print(f'image0 should be: {target0}.  random classifier guess image0 to be: {guess0}.')
    print('===')

    # test random digit classifier on the first 1000 images
    print('===')
    accuracy = test_digit_classifier(random_digit_classifier, 1000)
    print(f'random digit classifier: accuracy={accuracy*100}%.')
    print('===')

    # test average digit classifier on the first 1000 images
    print('===')
    calculate_average_digits()
    accuracy = test_digit_classifier(average_digit_classifier, 1000)
    print(f'average digit classifier: accuracy={accuracy*100}%.')
    print('===')



if __name__ == "__main__":
    main()
