from vectors import length


def secant_slope(f, xmin, xmax):
    return (f(xmax) - f(xmin)) / (xmax - xmin)


def approx_derivative(f, x, dx=1e-6):
    return secant_slope(f, x-dx, x+dx)


def approx_gradient3(f, x0, y0, z0, dx=1e-6):
    partial_x = approx_derivative(lambda x: f(x, y0, z0), x0, dx=dx)
    partial_y = approx_derivative(lambda y: f(x0, y, z0), y0, dx=dx)
    partial_z = approx_derivative(lambda z: f(x0, y0, z), z0, dx=dx)
    return (partial_x, partial_y, partial_z)


def gradient_descent3(f, xstart, ystart, zstart,
                      tolerance=1e-6, max_steps=1000):
    x = xstart
    y = ystart
    z = zstart
    grad = approx_gradient3(f, x, y, z)
    steps = 0
    while length(grad) > tolerance and steps < max_steps:
        x -= 0.01 * grad[0]
        y -= 0.01 * grad[1]
        z -= 0.01 * grad[2]
        grad = approx_gradient3(f, x, y, z)
        steps += 1
    print(f'gradient_descent3: max_steps = {max_steps}')
    print(f'gradient_descent3: converged or reached max_steps after {steps} steps')
    return x, y, z

