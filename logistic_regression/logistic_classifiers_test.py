from car_data import bmws, priuses
from scale import make_scale
from logistic_classifiers import compare_logistic_cost_functions
from test_classifier import test_classifier


def load_and_label_all_car_data():
    global all_car_data
    all_car_data = []
    for bmw in bmws:
        all_car_data.append((bmw.mileage, bmw.price, 1))
    for prius in priuses:
        all_car_data.append((prius.mileage, prius.price, 0))


def make_scaled_car_data():
    global scaled_car_data
    mileage_scale, mileage_unscale = make_scale([x[0] for x in all_car_data])
    price_scale, price_unscale = make_scale([x[1] for x in all_car_data])
    scaled_car_data = [(mileage_scale(mileage), price_scale(price), is_bmw)
                    for mileage, price, is_bmw in all_car_data]


def main():
    load_and_label_all_car_data()
    print('all_car_data:')
    print(all_car_data)

    # scale all_car_data where mileage and price values are btw 0 and 1
    make_scaled_car_data()
    print('scaled_car_data:')
    print(scaled_car_data)

    # a=0.35, b=1, c=0.56 => f(x, p) = 0.35 * x + 1 * p + 0.56
    compare_logistic_cost_functions(0.35, 1, 0.56, scaled_car_data)

    # a=1, b=1, c=1 => f(x, p) = 1 * x + 1 * p + 1
    compare_logistic_cost_functions(1, 1, 1, scaled_car_data)


if __name__ == "__main__":
    main()

