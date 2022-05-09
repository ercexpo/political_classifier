# Load the dataset into a pandas dataframe.
import pandas as pd

def get_data(csv_file):
    df = pd.read_csv(csv_file, delimiter=',', header=0).dropna()
    comments = df.title.values
    url_ids = df.min_id.values

    print('Number of inference sentences: {:,}\n'.format(df.shape[0]))
    return comments, url_ids
