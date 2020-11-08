import pandas as pd
from dfutil import normalize_df_strings, print_df_stats
from dfutil import get_encoded_df, get_correlated_df
from dfutil import split_df_to_csv_files


def main():
    print("Starting technical_approval_required ...")

    csv_file_name = 'orders_with_predicted_value.csv'
    raw_df = pd.read_csv(csv_file_name)
    normalize_df_strings(raw_df)
    print_df_stats(raw_df, 'raw_df')

    encoded_df = get_encoded_df(raw_df)
    print_df_stats(encoded_df, 'encoded_df')

    target_column_name = 'tech_approval_required'
    threshold = .1
    correlated_df= get_correlated_df(encoded_df, target_column_name, threshold)
    print_df_stats(correlated_df, 'correlated_df')

    train_df, val_df, test_df = split_df_to_csv_files(correlated_df)
    print_df_stats(train_df, 'train_df')
    print_df_stats(val_df, 'val_df')
    print_df_stats(test_df, 'test_df')


if __name__ == "__main__":
    main()

