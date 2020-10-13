from math import exp, log


def set_logistic_classifiers_data(data):
    global lc_data
    lc_data = data


def sigmoid(x):
    return 1 / (1+exp(-x))


def make_logistic(a,b,c):
    def l(x,p):
        return sigmoid(a*x + b*p - c)
    return l


# abs cost function: error = abs dist btw label and logistic function
def abs_point_cost(l, x, p, is_bmw):
    return abs(is_bmw - l(x, p))

def abs_logistic_cost(a, b, c):
    l = make_logistic(a, b, c)
    errors = [abs_point_cost(l, x, p, is_bmw)
              for x, p, is_bmw in lc_data]
    return sum(errors)


# log cost function: penalize wrong answer by using log function to amplify error
def log_point_cost(l, x, p, is_bmw): #1
    wrong = 1 - is_bmw
    return -log(abs(wrong - l(x, p)))

def log_logistic_cost(a, b, c):
    l = make_logistic(a, b, c)
    errors = [log_point_cost(l, x, p, is_bmw) #2
              for x, p, is_bmw in lc_data]
    return sum(errors)


def compare_logistic_cost_functions(a, b, c):
    abs_cost = abs_logistic_cost(a, b, c)
    log_cost = log_logistic_cost(a, b, c)

    print('===')
    print(f'a={a}, b={b}, c={c} => f(x, p) = {a} * x + {b} * p + {c}')
    print(f'wrong answer not amplified: abs_cost = {abs_cost}')
    print(f'wrong answer amplified by log: log_cost = {log_cost}')
    print('===')

