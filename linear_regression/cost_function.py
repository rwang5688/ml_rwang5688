def sum_squared_error(f, data):
    squared_errors = [(f(x) - y)**2 for (x,y) in data]
    return sum(squared_errors)

def coefficient_cost(a, b, data):
    def p(x):
        return a * x + b
    return sum_squared_error(p, data)

def scaled_cost_function(c, d, data):
    return coefficient_cost(0.5*c,50000*d, data)/1e13

