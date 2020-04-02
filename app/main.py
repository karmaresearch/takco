import sys, os, glob, json
import argh, tqdm

from data import read_t2d, kbs, Db
from stats import get_table_kblinks, get_kbinfo, get_novelty

def data():
    return

@argh.arg('kbid', choices=list(kbs))
def kblinks(
        data_file : 'Data file',
        kbdir : 'KB root directory', 
        kbid : 'Current KB id',
    ):
    
    kbpath = os.path.realpath(os.path.join(kbdir, *kbs[kbid]['path']))
    kb = Db(kbpath, **kbs[kbid])
    
    datasets = json.load(open(data_file))
    
    dataset_stats = {}
    for datasetname, tables in datasets.items():
        for table in tqdm.tqdm(tables, datasetname):
            kblinks = get_table_kblinks(kb, table)
            kblinks['novelty'] = get_novelty(kb, kblinks)
            table.setdefault('kblinks', {})[kbid] = kblinks
            
    with open(data_file, 'w') as fw:
        json.dump(datasets, fw)

if __name__ == '__main__':
    argh.dispatch_commands([data, kblinks])