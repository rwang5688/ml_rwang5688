from math import exp
from car_data import distinct_priuses, priuses, prius_mileage_price_data
from error_functions import sum_squared_errors
from gradient_descent import gradient_descent


# unscale linear coefficients
def unscale_linear_coefficients(c, d):
    a = 0.5 * c
    b = 50000 * d
    return a, b

# define unscaled linear coefficient cost function over car data
def linear_coefficient_cost_over_car_data(a, b):
    def p(x):
        return a * x + b
    return sum_squared_errors(p, prius_mileage_price_data)

# define scaled linear coefficient cost function over car data
def scaled_linear_coefficient_cost_over_car_data(c, d):
    a, b = unscale_linear_coefficients(c, d)
    return linear_coefficient_cost_over_car_data(a, b)/1e13


# unscale exponential coefficients
def unscale_exp_coefficients(s, t):
    q = 30000 * s
    r = 1e-4 * t
    return q, r

# define unscaled exponential coefficient cost function over car data
def exp_coefficient_cost_over_car_data(q, r):
    def f(x):
        return q * exp(r * x)
    return sum_squared_errors(f, prius_mileage_price_data)

# define scaled exponential coefficient cost fucntion over car data
def scaled_exp_coefficient_cost_over_car_data(s, t):
    q, r = unscale_exp_coefficients(s, t)
    return exp_coefficient_cost_over_car_data(q, r) / 1e11


def main():
    print('==')
    print('distinct_priuses:')
    print('==')
    print(distinct_priuses)
    print('==')
    print('priuses:')
    print('==')
    print(priuses)
    print('==')
    print('prius_mileage_price_data:')
    print('==')
    print(prius_mileage_price_data)
    print('==')

    print('==')
    print('starting gradient descent to find line of best fit ...')
    print('==')
    c, d = gradient_descent(scaled_linear_coefficient_cost_over_car_data, 0, 0)
    print(f'c = {c}, d = {d}')
    a, b = unscale_linear_coefficients(c, d)
    print(f'line of best fit: {a} * x + {b}')
    print('==')

    print('==')
    print('starting gradient descent to find exponential function of best fit ...')
    print('==')
    s, t = gradient_descent(scaled_exp_coefficient_cost_over_car_data, 0, 0)
    print('==')
    print(f's = {s}, t = {t}')
    q, r = unscale_exp_coefficients(s, t)
    print(f'exponential function of best fit: {q} * exp({r} * x)')
    print('==')


if __name__ == "__main__":
    main()

