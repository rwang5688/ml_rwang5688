import numpy as np
from sklearn import datasets


# initialize test set
def load_digits_and_test_set(test_start=0, test_count=1000):
    global digits
    global t_start, t_count, t_end
    global test_set_images, test_set_target
    digits = datasets.load_digits()
    t_images = len(digits.images)
    t_start = test_start
    t_count = test_count
    t_end = t_start + t_count
    if (t_images < t_end):
        t_end = t_images
    test_set_images = digits.images[t_start:t_end]
    test_set_target = digits.target[t_start:t_end]
    return len(test_set_images)


# test classifier over test set
def test_digit_classifier(classifier):
    digits = datasets.load_digits()
    correct = 0
    print(f'test classifier for {t_count} images: {t_start} to {t_end-1}.')
    for img, target in zip(test_set_images, test_set_target):
        v = np.matrix.flatten(img) / 15.
        output = classifier(v)
        answer = list(output).index(max(output))
        if answer == target:
            correct += 1
    return (correct/t_count)


# calculate total cost for classifier over test set
def y_vec(digit):
    return np.array([1 if i == digit else 0 for i in range(0,10)])


def cost_one(classifier,x,i):
    return sum([(classifier(x)[j] - y_vec(i)[j])**2 for j in range(10)])


def calculate_total_cost(classifier):
    digits = datasets.load_digits()
    print(f'calculate total cost for {t_count} images: {t_start} to {t_end-1}.')
    x = np.array([np.matrix.flatten(img) for img in test_set_images]) / 15.0
    y = test_set_target
    return sum([cost_one(classifier ,x[j], y[j]) for j in range(t_count)])/t_count

