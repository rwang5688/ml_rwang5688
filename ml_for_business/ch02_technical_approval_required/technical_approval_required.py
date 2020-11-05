import pandas as pd
from dfutil import print_df_stats, get_encoded_df, get_correlated_columns
from dfutil import split_correlated_df_to_csv_files


def main():
    print("Starting technical_approval_required ...")

    csv_file_name = 'orders_with_predicted_value.csv'
    raw_df = pd.read_csv(csv_file_name)

    df_name = 'raw_df'
    print_df_stats(raw_df, df_name)

    encoded_df = get_encoded_df(raw_df)

    df_name = 'encoded_df'
    print_df_stats(encoded_df, df_name)

    label_column_name = 'tech_approval_required'
    threshold = .1
    columns = get_correlated_columns(encoded_df, label_column_name, threshold)
    correlated_df = encoded_df[columns]

    df_name = 'correlated_df'
    print_df_stats(correlated_df, df_name)

    train_df, val_df, test_df = split_correlated_df_to_csv_files(correlated_df)

    df_name = 'train_df'
    print_df_stats(train_df, df_name)
    df_name = 'val_df'
    print_df_stats(val_df, df_name)
    df_name = 'test_df'
    print_df_stats(test_df, df_name)


if __name__ == "__main__":
    main()

