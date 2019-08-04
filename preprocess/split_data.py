import json
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = '/Users/jingjing/Data/arxiv_subject_classification/train.json'
# INPUT_FILE = '../data/train.json'


def read_data(file=INPUT_FILE):
    with open(file) as f:
        df = pd.DataFrame(json.load(f))
    return df


def filter_data(df):
    # remove duplicated titles
    df = df[~df.title.duplicated()]
    return df


def split_data(df, test_size=0.1):
    train, test = train_test_split(df, test_size=test_size, random_state=0, stratify=None)
    return train, test


def get_split_data(file=INPUT_FILE):
    df = filter_data(read_data(file))
    train, test = split_data(df)
    return train, test


if __name__ == '__main__':
    df_train, df_test = get_split_data()
    print(df_train.info())
    print(df_test.info())
