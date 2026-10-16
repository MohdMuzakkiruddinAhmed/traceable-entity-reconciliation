import argparse
from ..io import ROOT,read_json
from .runner import run

def main():
    parser=argparse.ArgumentParser(description='Exploratory KG-ERP improvements; original experiments preserved')
    parser.add_argument('command',choices=['run','explain'])
    parser.add_argument('--config',default=str(ROOT/'configs/improvements.yaml'))
    parser.add_argument('--run-id',default='improvements-v1')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--candidate-id')
    args=parser.parse_args()
    if args.command=='run':run(args.config,args.run_id)
    else:
        from pathlib import Path
        if Path(args.run_id).name!=args.run_id:raise ValueError('Invalid run ID')
        if not args.candidate_id:raise ValueError('--candidate-id is required for explanation')
        decisions=read_json(ROOT/f'results/{args.run_id}/wwi/seed_{args.seed}/decisions.json')
        decision=next((d for d in decisions if d['candidate_id']==args.candidate_id),None)
        if decision is None:raise ValueError('Unknown candidate ID')
        import json
        print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
