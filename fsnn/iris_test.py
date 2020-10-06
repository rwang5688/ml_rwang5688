# iris_test.py
# From Classic Computer Science Problems in Python Chapter 7
# Copyright 2018 David Kopec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# rwang5688@outlook.com: Refactored the original code for readability


import csv
from typing import List
from random import shuffle
from normalization import normalize_by_feature_scaling
from network import Network


SPECIES_TO_CLASSIFICATIONS = {
    'Iris-setosa': [1.0, 0.0, 0.0],
    'Iris-versicolor': [0.0, 1.0, 0.0],
    'Iris-virginica': [0.0, 0.0, 1.0]
}


def iris_interpret_output(output: List[float]) -> str:
    if max(output) == output[0]:
        return "Iris-setosa"
    elif max(output) == output[1]:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"


def main():
    iris_parameters: List[List[float]] = []
    iris_classifications: List[List[float]] = []
    iris_species: List[str] = []
    with open('iris.csv', mode='r') as iris_file:
        irises: List = list(csv.reader(iris_file))
        # shuffle data set in random order
        shuffle(irises)
        for iris in irises:
            parameters: List[float] = [float(n) for n in iris[0:4]]
            iris_parameters.append(parameters)
            species: str = iris[4]
            iris_classifications.append(SPECIES_TO_CLASSIFICATIONS[species])
            iris_species.append(species)
    normalize_by_feature_scaling(iris_parameters)

    # for this test:
    # create a network with 3 layers
    # input layer: 4 neurons
    # hidden layer: 6 neurons
    # output layer: 3 neurons
    # learning rate: 0.3
    iris_network: Network = Network([4, 6, 3], 0.3)

    # training set: first 140 (out of 150) irises in the shuffled data set
    iris_trainers: List[List[float]] = iris_parameters[0:140]
    iris_trainers_corrects: List[List[float]] = iris_classifications[0:140]
    for i in range(140):
        print(f'training set {i}: parameters: {iris_trainers[i]}; classifications {iris_trainers_corrects[i]}')

    # training iterations: 50
    for i in range(50):
        print(f'training iteration {i} ...')
        iris_network.train(iris_trainers, iris_trainers_corrects)

    # validation set: last 10 (out of 150) of the irises in the shuffled data set
    iris_testers: List[List[float]] = iris_parameters[140:150]
    iris_testers_corrects: List[str] = iris_species[140:150]
    for i in range(10):
        print(f'training set {i}: parameters: {iris_testers[i]}; classifications {iris_testers_corrects[i]}')

    print('validation:')
    iris_results = iris_network.validate(iris_testers, iris_testers_corrects, iris_interpret_output)
    print(f'{iris_results[0]} correct of {iris_results[1]} = {iris_results[2] * 100}%')


if __name__ == "__main__":
    main()

