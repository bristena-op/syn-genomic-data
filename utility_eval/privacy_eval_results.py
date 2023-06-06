from glob import glob
import json
from os import path


def load_results(directory, dataname, attack='MIA'):
    files = glob(path.join(directory, f'{dataname}*.json'))
    res_lsit = []

    for fname in files:
        gm = fname.split('/')[-1].split(attack)[0].split(dataname)[-1]
        with open(fname) as file:
            rd = json.load(file)

        rdf = parse_results_mia(rd)
        rdf['GenerativeModel'] = gm
        rdf['Dataset'] = dname
        resList.append(rdf)