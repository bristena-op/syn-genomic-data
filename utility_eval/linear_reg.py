from sklearn.linear_model import LinearRegression
import pandas as pd


def lg(syn_data, r_data):
    X_train = syn_data[:-1]
    X_test = r_data[:-1]



if __name__ == '__main__':
    dataset = 'CHB'
    data_file = f'/home/bristena/syn_genomics/datasets/chr13/small_{dataset}.chr13.hap'
    dat = pd.read_csv(data_file, sep='\t', header=None)