import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_df_strings(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')


def print_df_stats(df, df_name):
    print("==")
    print("print_df_stat: %s" % df_name)
    print("First five rows in dataset:")
    print(df.head())
    print("Number of rows in dataset: %d." % df.shape[0])
    print("Number of columns in dataset: %d." % df.shape[1])
    print("Distribution of values columns[0]:")
    print(df[df.columns[0]].value_counts())
    print("Number of null values in each column:")
    print(df.isnull().sum())
    print("==")


def get_encoded_df(df):
    encoded_df = pd.get_dummies(df)
    return encoded_df


def get_correlated_df(df, target_column_name, threshold):
    correlated_columns = df.corr()[target_column_name].abs()
    correlated_columns_gt_threshold_indexes = correlated_columns[correlated_columns > threshold].index

    # debug: print columns with correlation > threshold
    correlated_columns_gt_threshold = correlated_columns.filter(correlated_columns_gt_threshold_indexes)
    print("==")
    print("get_correlated_df:")
    print("columns with correlation > %2.1f percent vs target column '%s':" % (threshold*100, target_column_name))
    print(correlated_columns_gt_threshold)
    print("==")

    correlated_df = df[correlated_columns_gt_threshold_indexes]
    return correlated_df


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

