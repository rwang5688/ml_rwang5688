# wine_test.py
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
from normalize import normalize_by_feature_scaling
from network import Network


SPECIES_TO_CLASSIFICATIONS = {
    1: [1.0, 0.0, 0.0],
    2: [0.0, 1.0, 0.0],
    3: [0.0, 0.0, 1.0]
}


def wine_interpret_output(output: List[float]) -> int:
    if max(output) == output[0]:
        return 1
    elif max(output) == output[1]:
        return 2
    else:
        return 3


def main():
    wine_parameters: List[List[float]] = []
    wine_classifications: List[List[float]] = []
    wine_species: List[int] = []
    with open('wine.csv', mode='r') as wine_file:
        wines: List = list(csv.reader(wine_file, quoting=csv.QUOTE_NONNUMERIC))
        # data set in random order
        shuffle(wines)
        for wine in wines:
            parameters: List[float] = [float(n) for n in wine[1:14]]
            wine_parameters.append(parameters)
            species: int = int(wine[0])
            wine_classifications.append(SPECIES_TO_CLASSIFICATIONS[species])
            wine_species.append(species)
    normalize_by_feature_scaling(wine_parameters)

    # for this test:
    # create a network with 3 layers
    # input layer: 13 neurons
    # hidden layer: 7 neurons
    # output layer: 3 neurons
    # learning rate: 0.9
    wine_network: Network = Network([13, 7, 3], 0.9)

    # training set: first 150 (out of 178) wines in the shuffled data set
    wine_trainers: List[List[float]] = wine_parameters[0:150]
    wine_trainers_corrects: List[List[float]] = wine_classifications[0:150]
    for i in range(150):
        print(f'training set {i}: parameters: {wine_trainers[i]}; classifications {wine_trainers_corrects[i]}')

    # training iterations: 10
    for i in range(10):
        print(f'training iteration {i} ...')
        wine_network.train(wine_trainers, wine_trainers_corrects)

    # validation set: last 28 (out of 178) of the wines in the shuffled data set
    wine_testers: List[List[float]] = wine_parameters[150:178]
    wine_testers_corrects: List[int] = wine_species[150:178]
    for i in range(28):
        print(f'training set {i}: parameters: {wine_testers[i]}; classifications {wine_testers_corrects[i]}')

    print('validation:')
    wine_results = wine_network.validate(wine_testers, wine_testers_corrects, wine_interpret_output)
    print(f'{wine_results[0]} correct of {wine_results[1]} = {wine_results[2] * 100}%')


if __name__ == "__main__":
    main()
