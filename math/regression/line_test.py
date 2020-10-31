from line_data import line_data
from error_functions import sum_squared_errors
from gradient_descent2 import gradient_descent2


# define linear cost function over line data
def linear_cost_over_line_data(a, b):
    def f(x):
        return a*x+b
    return sum_squared_errors(f, line_data)


def main():
    print('==')
    print('line_data:')
    print('==')
    print(line_data)
    print('==')

    n = len(line_data)
    print('==')
    print(f'starting gradient descent to find line of best fit over {n} points ...')
    print('==')
    a, b = gradient_descent2(linear_cost_over_line_data, 0, 0)
    print('==')
    print(f'line of best fit: {a} * x + {b}')
    print('==')


if __name__ == "__main__":
    main()

