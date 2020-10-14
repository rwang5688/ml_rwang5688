def print_array(a):
    print('===')
    print(f'a: {a}')
    print(f'a[:-1]: {a[:-1]}')
    print(f'a[1:]: {a[1:]}')
    print('===')


def main():
    print_array([0, 1, 2, 3, 4, 5])
    print_array([2,3])
    print_array([64, 16, 10])


if __name__ == "__main__":
    main()

