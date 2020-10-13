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
    slope = -(a/b)
    intercept = c/b

    print('===')
    print(f'a={a}, b={b}, c={c} =>')
    print(f'f(x, p) = {a} * x + {b} * p - {c}, or')
    print(f'for scaled (x, p): p = {slope} * x + {intercept}')
    print(f'wrong answer not amplified: abs_cost = {abs_cost}')
    print(f'wrong answer amplified by log: log_cost = {log_cost}')
    print('===')


# manual logistic classifier
def manual_logistic_classifier(x,p):
    l = make_logistic(0.35, 1, 0.56)
    if l(x,p) > 0.5:
        return 1
    else:
        return 0


# best logistic classifier based on best (a, b, c)
def set_best_a_b_c(a, b, c):
    global best_a, best_b, best_c
    best_a = a
    best_b = b
    best_c = c


def best_logistic_classifier(x,p):
    l = make_logistic(best_a, best_b, best_c)
    if l(x,p) > 0.5:
        return 1
    else:
        return 0

