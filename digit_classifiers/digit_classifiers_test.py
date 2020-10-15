import numpy as np
from sklearn import datasets
from digit_classifiers import load_digits_and_training_set
from digit_classifiers import random_digit_classifier
from digit_classifiers import calculate_average_digits, average_digit_classifier
from digit_classifiers import set_layer_sizes, random_mlp_digit_classifier
from digit_classifiers import train_sklearn_mlp_digit_classifier, sklearn_mlp_digit_classifier
from test_digit_classifier import load_digits_and_test_set
from test_digit_classifier import test_digit_classifier, calculate_total_cost


def setup():
    global digits
    global image0, flattened_image0, scaled_and_flattened_image0, target0
    global layer_sizes

    print('==')
    print('begin setup:')

    # load digits, training set, test set
    digits = datasets.load_digits()
    training_count = load_digits_and_training_set(training_start=1000, training_count=1000)
    test_count = load_digits_and_test_set(test_start=0, test_count=1000)
    print(f'digit dataset has {len(digits.images)} images:')
    print(f'training set has {training_count} images.')
    print(f'test set has {test_count} images.')

    # get image0, which happens to be digit '0'
    image0 = digits.images[0]
    print('image0:')
    print(image0)

    flattened_image0 = np.matrix.flatten(image0)
    print('falttened image0:')
    print(flattened_image0)

    scaled_and_flattened_image0 = flattened_image0 / 15
    print('scaled_and_flattened_image0:')
    print(scaled_and_flattened_image0)

    target0 = digits.target[0]
    print(f'image0 (target0) should be: {target0}.')

    # average_digit_classifier: calculate average from image1000 onward
    calculate_average_digits()

    # random_mlp_digit_classifier: set layer sizes
    layer_sizes = [64, 16, 10]
    set_layer_sizes(layer_sizes)

    # sklearn_mlp_digit_classifier: train from image1000 onward
    train_sklearn_mlp_digit_classifier()

    print('end setup.')
    print('==')


def exercise_digit_classifier(classifier, classifier_name, test_start=0, test_count=1000):
    # use classifier on image0
    print('===')
    print(f'begin exercise: {classifier_name}')

    print('---')
    result0 = classifier(scaled_and_flattened_image0)
    print(f'{classifier_name} result:')
    print(result0)
    guess0 = list(result0).index(max(result0))
    print(f'image0 should be: {target0}.  {classifier_name} guess image0 to be: {guess0}.')
    print('---')

    # test classifier over the test set
    print('---')
    accuracy = test_digit_classifier(classifier)
    print(f'{classifier_name}: accuracy={accuracy*100}%.')
    print('---')

    # calculate total cost for classifier over the test set
    print('---')
    total_cost = calculate_total_cost(classifier)
    print(f'{classifier_name}: total cost={total_cost}.')
    print('---')

    print(f'end exercise: {classifier_name}.')
    print('===')


def main():
    setup()

    # exercise random digit classifier
    exercise_digit_classifier(random_digit_classifier,
                                "random_digit_classifier",
                                test_start=0, test_count=1000)

    # exercise average digit classifier on image0
    exercise_digit_classifier(average_digit_classifier,
                                "average_digit_classifier",
                                test_start=0, test_count=1000)

    # exercise random MLP digit classifier
    exercise_digit_classifier(random_mlp_digit_classifier,
                                "random_mlp_digit_classifier",
                                test_start=0, test_count=1000)

    # exercise sklearn MLP digit classifier
    exercise_digit_classifier(sklearn_mlp_digit_classifier,
                                "sklearn_mlp_digit_classifier",
                                test_start=0, test_count=1000)


if __name__ == "__main__":
    main()

