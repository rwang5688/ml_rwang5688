import pandas as pd
from dfutil import normalize_df_strings, print_df_stats
from dfutil import np_split_df


def main():
    car_df = pd.read_csv('data.csv')
    normalize_df_strings(car_df)
    print_df_stats(car_df, 'car_df')

    train_df, val_df, test_df = np_split_df(car_df)
    print_df_stats(train_df, 'train_df')
    print_df_stats(val_df, 'val_df')
    print_df_stats(test_df, 'test_df')


if __name__ == "__main__":
    main()

