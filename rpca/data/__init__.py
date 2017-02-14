import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_sleep():
    folder = os.path.dirname(__file__)
    data_raw = pd.read_csv(os.path.join(folder, 'sleep.txt'), sep='\t', index_col=0)
    data_dropped = data_raw[['BodyWt', 'BrainWt', 'LifeSpan', 'Gestation']].dropna()
    return MinMaxScaler((0, 100)).fit_transform(data_dropped)
