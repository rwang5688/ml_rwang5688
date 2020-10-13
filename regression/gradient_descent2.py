from vectors import length


def secant_slope(f, xmin, xmax):
    return (f(xmax) - f(xmin)) / (xmax - xmin)


def approx_derivative(f, x, dx=1e-6):
    return secant_slope(f, x-dx, x+dx)


def approx_gradient2(f, x0, y0, dx=1e-6):
    partial_x = approx_derivative(lambda x:f(x, y0), x0, dx=dx)
    partial_y = approx_derivative(lambda y:f(x0, y), y0, dx=dx)
    return (partial_x, partial_y)


def gradient_descent2(f, xstart, ystart, tolerance=1e-6):
    i = 0
    x = xstart
    y = ystart
    grad = approx_gradient2(f, x, y)
    while length(grad) > tolerance:
        i = i + 1
        x -= 0.01 * grad[0]
        y -= 0.01 * grad[1]
        grad = approx_gradient2(f, x, y)
    print(f'gradient_descent: converged after {i} iterations')
    return x, y

