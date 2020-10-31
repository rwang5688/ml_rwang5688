from car_data import bmws, priuses
from scale import make_scale
from logistic_classifiers import set_logistic_classifiers_data
from logistic_classifiers import compare_logistic_cost_functions
from logistic_classifiers import log_logistic_cost
from gradient_descent3 import gradient_descent3
from logistic_classifiers import manual_logistic_classifier
from logistic_classifiers import set_best_a_b_c, best_logistic_classifier
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


def run_gradient_descent3(f, xstart, ystart, zstart,
                      tolerance=1e-6, max_steps=1000):
    print('===')
    a, b, c = gradient_descent3(f, xstart, ystart, zstart, tolerance, max_steps)
    log_cost = log_logistic_cost(a, b, c)
    slope = -(a/b)
    intercept = c/b
    print(f'a={a}, b={b}, c={c} =>')
    print(f'f(x, p) = {a} * x + {b} * p - {c}, or')
    print(f'for scaled (x, p): p = {slope} * x + {intercept}')
    print(f'log_cost = {log_cost}')
    print('===')

    return a, b, c


def main():
    load_and_label_all_car_data()
    print('all_car_data:')
    print(all_car_data)

    # scale all_car_data where mileage and price values are btw 0 and 1
    make_scaled_car_data()
    print('scaled_car_data:')
    print(scaled_car_data)

    # set data to be used by logistic classifiers
    set_logistic_classifiers_data(scaled_car_data)

    # a=1, b=1, c=1 => f(x, p) = 1 * x + 1 * p - 1
    compare_logistic_cost_functions(1, 1, 1)

    # a=0.35, b=1, c=0.56 => f(x, p) = 0.35 * x + 1 * p - 0.56
    compare_logistic_cost_functions(0.35, 1, 0.56)

    # run gradient descent to find logistic classifier of best fit
    a, b, c = run_gradient_descent3(log_logistic_cost, 1, 1, 1, max_steps=100)
    a, b, c = run_gradient_descent3(log_logistic_cost, 1, 1, 1, max_steps=1000)
    a, b, c = run_gradient_descent3(log_logistic_cost, 1, 1, 1, max_steps=10000)

    # compare manual vs. best logistic classifier
    print('===')
    accuracy = test_classifier(manual_logistic_classifier, scaled_car_data, True)
    print(f'manual_logistic_classifier accuracy={accuracy*100}%')
    print(f'where decision boundary is: 0.35 * x + 1 * p - 0.56')
    print('===')

    print('===')
    set_best_a_b_c(a, b, c)
    accuracy = test_classifier(best_logistic_classifier, scaled_car_data, True)
    print(f'best_logistic_classifier accuracy={accuracy*100}%')
    print(f'where decision boundary is: {a/b} * x + 1 * p - {c/b}')
    print('===')



if __name__ == "__main__":
    main()

