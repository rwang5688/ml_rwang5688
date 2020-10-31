from gradient_descent import gradient_descent


def sum_squares(*v):
    return sum([(x-1)**2 for x in v])


def run_gradient_descent(f, v_start, dx=1e-6, max_steps=1000):
    print('===')
    v_min = gradient_descent(f, v_start, dx, max_steps)
    print(f'v_start: {v_start}')
    print(f'v_min: {v_min}')
    print("===")

    return v_min


def main():
    v_start = [2,2,2,2,2,2,2,2]
    dx = 1e-6
    v_min = run_gradient_descent(sum_squares, v_start, dx, 100)
    v_min = run_gradient_descent(sum_squares, v_start, dx, 200)
    v_min = run_gradient_descent(sum_squares, v_start, dx, 400)
    v_min = run_gradient_descent(sum_squares, v_start, dx, 800)


if __name__ == "__main__":
    main()

