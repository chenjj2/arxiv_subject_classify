import json
from sklearn.model_selection import train_test_split

INPUT_FILE = '~/Data/arxiv_subject_classification/train.json'
DATA = json.load(open(INPUT_FILE))


def split(data=DATA):
    