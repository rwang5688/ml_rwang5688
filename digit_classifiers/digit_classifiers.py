import numpy as np
from sklearn import datasets


# random classifier: return an array of 10 probabilities between 0 and 1
# that represent the probabilities that the image is digit 0 .. 9
def random_digit_classifier(v):
    return np.random.rand(10)


# average classifer: return an array of dot products btw input vector and avg image vector
# avg image vector:
#   for each digit from 0 .. 9
#       calculate an avg image from image1000 onward
def average_img(i):
    imgs = [img for img,target in zip(digits.images[1000:], digits.target[1000:]) if target == i]
    return sum(imgs) / len(imgs)


def calculate_average_digits():
    global digits
    global avg_digits
    digits = datasets.load_digits()
    avg_digits = [np.matrix.flatten(average_img(i)) for i in range(10)]


def average_digit_classifier(v):
    return [np.dot(v, avg_digits[i]) for i in range(10)]

