# Load the dataset into a pandas dataframe.
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(csv_file):
    df = pd.read_csv(csv_file, delimiter=',', header=0).dropna()

    labels = df.label.values
    comments = df.title.values

    train_comments, val_comments, train_labels, val_labels = train_test_split(comments, labels, test_size = .1)

    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    return train_comments, train_labels, val_comments, val_labels
