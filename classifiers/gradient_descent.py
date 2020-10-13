from vectors import length


def secant_slope(f, xmin, xmax):
    return (f(xmax) - f(xmin)) / (xmax - xmin)


def approx_derivative(f, x, dx=1e-6):
    return secant_slope(f, x-dx, x+dx)


def partial_derivative(f, i, v, dx=1e-6):
    def cross_section(x):
        arg = [(vj if j != i else x) for j, vj in enumerate(v)]
        return f(*arg)
    return approx_derivative(cross_section, v[i], dx)


def approx_gradient(f, v, dx=1e-6):
    return [partial_derivative(f, i, v, dx) for i in range(0, len(v))]


def gradient_descent(f, vstart, dx=1e-6, max_steps=1000):
    v = vstart
    grad = approx_gradient(f, v, dx)
    steps = 0
    while length(grad) > dx and steps < max_steps:
        v = [(vi - 0.01 * dvi) for vi, dvi in zip(v, grad)]
        grad = approx_gradient(f, v, dx)
        steps += 1
    print(f'gradient_descent: max_steps = {max_steps}')
    print(f'gradient_descent: converged or reached max_steps after {steps} steps')
    return v

