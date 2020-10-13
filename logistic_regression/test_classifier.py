def test_classifier(classifier, data, verbose=False): #1
    true_positives = 0 #2
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    num_data_points = len(data)
    for mileage, price, is_bmw in data:
        predicted = classifier(mileage,price)
        if predicted and is_bmw: #3
            true_positives += 1
        elif not predicted and not is_bmw:
            true_negatives += 1
        elif predicted and not is_bmw:
            false_positives += 1
        elif not predicted and is_bmw:
            false_negatives += 1

    if verbose:
        print("true positives: %f" % true_positives) #4
        print("true negatives: %f" % true_negatives)
        print("false positives: %f" % false_positives)
        print("false negatives: %f" % false_negatives)
        print("num data points: %f" % num_data_points)

    return (true_positives + true_negatives) / num_data_points #5

