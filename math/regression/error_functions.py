def sum_squared_errors(f, data):
    squared_errors = [(f(x) - y)**2 for (x,y) in data]
    return sum(squared_errors)

