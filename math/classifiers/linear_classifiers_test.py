from car_data import bmws, priuses
from linear_classifiers import bmw_finder_price_gt_25k
from linear_classifiers import bmw_finder_price_gt_20k
from linear_classifiers import bmw_finder_price_gt_cutoff_price
from linear_classifiers import bmw_finder_decision_boundary
from test_classifier import test_classifier


def load_and_label_all_car_data():
    global all_car_data
    all_car_data = []
    for bmw in bmws:
        all_car_data.append((bmw.mileage, bmw.price, 1))
    for prius in priuses:
        all_car_data.append((prius.mileage, prius.price, 0))


def cutoff_accuracy(cutoff_price):
    c = bmw_finder_price_gt_cutoff_price(cutoff_price)
    return test_classifier(c, all_car_data)


def main():
    load_and_label_all_car_data()
    print('all_car_data:')
    print(all_car_data)

    # constant decision boundaries: cutoff price = 25k, 20k, most accurate cutoff
    print('===')
    accuracy = test_classifier(bmw_finder_price_gt_25k, all_car_data, True)
    print(f'bmw_finder_price_gt_25k accuracy={accuracy*100}%')
    print('===')

    print('===')
    accuracy = test_classifier(bmw_finder_price_gt_20k, all_car_data, True)
    print(f'bmw_finder_price_gt_20k accuracy={accuracy*100}%')
    print('===')

    print('===')
    all_prices = [price for (mileage, price, is_bmw) in all_car_data]
    most_accurate_cutoff_price = max(all_prices, key=cutoff_accuracy)
    most_accurate_cutoff_price_classifier = bmw_finder_price_gt_cutoff_price(most_accurate_cutoff_price)
    best_accuracy = test_classifier(most_accurate_cutoff_price_classifier, all_car_data, True)
    print(f'most accurate cutoff price is {most_accurate_cutoff_price} at accuracy={best_accuracy*100}%')
    print('===')

    # linear decision boundary: price = 21000 - 0.07 * mileage
    print('===')
    accuracy = test_classifier(bmw_finder_decision_boundary, all_car_data, True)
    print(f'bmw_finder_decision_boundary accuracy={accuracy*100}%')
    print('where decision boundary is: price = 21k - 0.07 * mileage')
    print('===')


if __name__ == "__main__":
    main()

