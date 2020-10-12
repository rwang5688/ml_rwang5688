from car_data import bmws, priuses
from scale import make_scale


def load_all_car_data():
    global all_car_data
    all_car_data = []
    for bmw in bmws:
        all_car_data.append((bmw.mileage, bmw.price, 1))
    for prius in priuses:
        all_car_data.append((prius.mileage, prius.price, 0))


def main():
    load_all_car_data()
    print('all_car_data:')
    print(all_car_data)

    # create a scaled version of all_car_data where mileage and price values are btw 0 and 1
    mileage_scale, mileage_unscale = make_scale([x[0] for x in all_car_data])
    price_scale, price_unscale = make_scale([x[1] for x in all_car_data])
    scaled_car_data = [(mileage_scale(mileage), price_scale(price), is_bmw)
                    for mileage, price, is_bmw in all_car_data]
    print('scaled_car_data:')
    print(scaled_car_data)


if __name__ == "__main__":
    main()

