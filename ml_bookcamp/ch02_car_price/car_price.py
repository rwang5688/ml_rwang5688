import pandas as pd
from dfutil import normalize_df_strings, print_df_stats
import numpy as np


def main():
    car_df = pd.read_csv('data.csv')
    normalize_df_strings(car_df)
    print_df_stats(car_df, 'car_df')


if __name__ == "__main__":
    main()

