import pandas as pd


def main():
    print("Starting technical_approval_required ...")

    df = pd.read_csv('orders_with_predicted_value.csv')

    print('First five rows in dataset:')
    print(df.head())
    print(f'Number of rows in dataset: {df.shape[0]}')
    print(df[df.columns[0]].value_counts())


if __name__ == "__main__":
    main()

