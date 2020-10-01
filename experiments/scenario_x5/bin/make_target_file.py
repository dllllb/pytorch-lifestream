import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/clients.csv')
    df['age'] = np.where(
        (df['age'] < 10) | (df['age'] > 90), -1, np.where(
            df['age'] < 35, 0, np.where(
                df['age'] < 45, 1, np.where(
                    df['age'] < 60, 2, 3))))
    df = df[df['age'].ge(0)]
    df.to_csv('data/clients_target.csv', index=False)
