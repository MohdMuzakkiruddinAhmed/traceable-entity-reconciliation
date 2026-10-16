import argparse
from pathlib import Path
from kgerp.io import ROOT
from kgerp.expanded.sources import prepare

parser=argparse.ArgumentParser()
parser.add_argument('--run-id',default='all-datasets-v1')
parser.add_argument('--datasets',nargs='+',default=['Olist','DataCo','BPI2019'])
args=parser.parse_args()
if Path(args.run_id).name!=args.run_id:raise ValueError('Invalid run ID')
for dataset in args.datasets:
    out=ROOT/'results'/args.run_id/dataset
    out.mkdir(parents=True,exist_ok=True)
    prepare(dataset,out)
    print('Full E1 source import complete: '+dataset,flush=True)
