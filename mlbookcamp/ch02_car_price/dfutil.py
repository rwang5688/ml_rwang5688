import pandas as pd
from sklearn.model_selection import train_test_split


def print_df_stats(df, df_name):
    print("==")
    print(f"print_df_stat: {df_name}")
    print("First five rows in dataset:")
    print(df.head())
    print(f"Number of rows in dataset: {df.shape[0]}.")
    print(f"Number of columns in dataset: {df.shape[1]}.")
    print("Distribution of values columns[0]:")
    print(df[df.columns[0]].value_counts())
    print("==")


def get_encoded_df(df):
    encoded_df = pd.get_dummies(df)
    return encoded_df


def get_correlated_columns_gt_threshold_indexes(df, label_column_name, threshold):
    correlated_columns = df.corr()[label_column_name].abs()
    correlated_columns_gt_threshold_indexes = correlated_columns[correlated_columns > threshold].index
    correlated_columns_gt_threshold = correlated_columns.filter(correlated_columns_gt_threshold_indexes)
    print("==")
    print("get_correlated_columns:")
    print(f"columns with correlation > {threshold*100}% vs label column '{label_column_name}':")
    print(correlated_columns_gt_threshold)
    print("==")
    return correlated_columns_gt_threshold_indexes


def write_df_to_csv_file(df, csv_file_name):
    df_data = df.to_csv(None, header=False, index=False).encode()
    with open(csv_file_name, 'wb') as f:
        f.write(df_data)


def split_df_to_csv_files(df):
    train_df, val_and_test_data = train_test_split(df, test_size=0.3, random_state=0)
    val_df, test_df = train_test_split(val_and_test_data, test_size=0.333, random_state=0)

    csv_file_name = 'train.csv'
    write_df_to_csv_file(train_df, csv_file_name)

    csv_file_name = 'val.csv'
    write_df_to_csv_file(val_df, csv_file_name)

    csv_file_name = 'test.csv'
    write_df_to_csv_file(val_df, csv_file_name)

    return train_df, val_df, test_df

