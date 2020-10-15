import numpy as np
from sklearn import datasets
from RandomMLP import RandomMLP
from sklearn.neural_network import MLPClassifier


# initialize training set
def load_digits_and_training_set(training_start=1000, training_count=1000):
    global digits
    global t_start, t_count, t_end
    global training_set_images, training_set_target
    digits = datasets.load_digits()
    t_images = len(digits.images)
    t_start = training_start
    t_count = training_count
    t_end = t_start + t_count
    if (t_images < t_end):
        t_end = t_images
    training_set_images = digits.images[t_start:t_end]
    training_set_target = digits.target[t_start:t_end]
    return len(training_set_images)


# random classifier: return an array of 10 probabilities between 0 and 1
# that represent the probabilities that the image is digit 0 .. 9
def random_digit_classifier(v):
    return np.random.rand(10)


# average classifer: return an array of dot products btw input vector and avg image vector
# avg image vector:
#   for each digit from 0 .. 9
#       "train" from image1000 onward: calculate an avg image from image1000 onward
def average_img(i):
    x = training_set_images
    y = training_set_target
    imgs = [img for img, target in zip(x, y) if target == i]
    return sum(imgs) / len(imgs)


def calculate_average_digits():
    global avg_digits
    print(f'calculate average digits over {len(training_set_images)} images.')
    avg_digits = [np.matrix.flatten(average_img(i)) for i in range(10)]


def average_digit_classifier(v):
    dot_products = [np.dot(v, avg_digits[i]) for i in range(10)]
    results = dot_products / max(dot_products)
    return results


# random MLP classifier: use a MLP with randomly generated weights and biases
def set_layer_sizes(lsizes):
    global layer_sizes
    layer_sizes = lsizes
    print(f'layer_sizes: {layer_sizes}')
    print(f'layer_sizes[:-1]: {layer_sizes[:-1]}')
    print(f'layer_sizes[1:]: {layer_sizes[1:]}')


def random_mlp_digit_classifier(v):
    nn = RandomMLP(layer_sizes)
    return nn.evaluate(v)


# sklearn MLP classifer: use a MLP that has been trained on image1000 onward
def train_sklearn_mlp_digit_classifier():
    global sklearn_mlp
    print(f'train sk MLP classifier over {len(training_set_images)} images.')
    x = np.array([np.matrix.flatten(img) for img in training_set_images]) / 15.0
    y = training_set_target

    sklearn_mlp = MLPClassifier(hidden_layer_sizes=(16,),
                                activation='logistic',
                                max_iter=100,
                                verbose=10,
                                random_state=1,
                                learning_rate_init=.1)
    sklearn_mlp.fit(x,y)


def sklearn_mlp_digit_classifier(v):
    return sklearn_mlp._predict([v])[0]

