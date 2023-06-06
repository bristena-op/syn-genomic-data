import json
import pandas as pd
import numpy as np

def metadata_hap(pos, data_file):
    mdata = {}
    mdata['columns'] = []
    for item in pos:
        it = f'{item}'
        mdata['columns'].append({
            'name': it,
            'type': "categorical",
            'size': '2',
            'i2s': [0,1]
        })
    m_data_file= data_file[0:-3]+'json'
    with open(m_data_file , 'w') as outfile:
        json.dump(mdata, outfile)
if __name__ == '__main__':
    dataset = 'CHB'
    data_file = f'../datasets/chr13/small_{dataset}.chr13.hap'
    dat = pd.read_csv(data_file,sep= '\t', header=None)
    # import pdb; pdb.set_trace()

    dat = dat.to_numpy()
    pos = dat[:, 0]
    dat = np.transpose(dat[:, 1:])
    metadata_hap(pos, data_file)
