import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import traceback
import yaml
from .io import ROOT,digest,write_json

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value);stream.flush()
    def flush(self):
        for stream in self.streams:stream.flush()

def run(config_path,run_id=None):
    from .ingest import load_tables
    from .evaluations import environment,e1,e2_seed,e3,e4,e5,wdc_evaluate,plots
    config=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    run_id=run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if Path(run_id).name!=run_id or run_id in {'.','..'}:raise ValueError('Invalid run ID')
    out=ROOT/'results'/run_id
    out.mkdir(parents=True,exist_ok=False)  # preserve immutable prior results
    (out/'config_snapshot.yaml').write_text(Path(config_path).read_text(encoding='utf-8'),encoding='utf-8')
    code_hashes={str(p.relative_to(ROOT)):digest(p) for folder in ['src','scripts','tests'] for p in sorted((ROOT/folder).rglob('*.py'))}
    input_manifests={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'data/manifests').glob('*')) if p.is_file()}
    write_json(out/'environment.json',environment())
    write_json(out/'run_manifest.json',{'status':'running','code_sha256':code_hashes,'config_sha256':digest(config_path),'scope':config['scope']})
    log=(out/'run.log').open('w',encoding='utf-8')
    original=sys.stdout;sys.stdout=Tee(original,log)
    completed=[]
    try:
        tables=load_tables()
        e1(tables,out);completed.append('E1')
        e2rows,e3rows,e4rows=[],[],[]
        bundle=None
        for seed in config['seeds']:
            bundle=e2_seed(tables,config,seed,out);e2rows.extend(bundle['metrics'])
            e3rows.extend(e3(bundle,seed,out));e4rows.extend(e4(bundle,seed,out,config))
        completed.extend(['E2_WWI','E3','E4'])
        wdcrows=wdc_evaluate(config['wdc'],out);completed.append('E2_WDC')
        e5rows=e5(bundle,out,config);completed.append('E5')
        plots(out,e2rows,e3rows,e4rows,e5rows,wdcrows)
        write_json(out/'run_manifest.json',{'status':'completed','completed':completed,'code_sha256':code_hashes,
                   'input_manifest_sha256':input_manifests,'requirements_lock_sha256':digest(ROOT/'requirements.lock'),
                   'config_sha256':digest(config_path),'scope':config['scope'],'claims':'exploratory; inspect measured null/negative findings',
                   'result_sha256':{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name not in {'run_manifest.json','run.log'}}})
        write_json(ROOT/'results/latest.json',{'run_id':run_id,'status':'completed'})
        print(f'Completed two-dataset E1–E5 run: {out}',flush=True)
    except Exception as exc:
        traceback.print_exc(file=log)
        write_json(out/'run_manifest.json',{'status':'failed','completed':completed,'error':str(exc),'code_sha256':code_hashes,'config_sha256':digest(config_path)})
        raise
    finally:
        sys.stdout=original;log.close()
    return out

def main():
    parser=argparse.ArgumentParser(description='KG-ERP two-dataset research pipeline')
    parser.add_argument('command',choices=['extract','profile','run','wdc','status'])
    parser.add_argument('--config',default=str(ROOT/'configs/eval_protocol.yaml'))
    parser.add_argument('--run-id')
    args=parser.parse_args()
    if args.command=='extract':
        from .ingest import extract
        extract()
    elif args.command=='profile':
        from .ingest import profile
        print(json.dumps(profile(),indent=2))
    elif args.command=='run':run(args.config,args.run_id)
    elif args.command=='wdc':
        from .evaluations import wdc_evaluate
        config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        out=ROOT/'results'/('wdc_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        out.mkdir(parents=True,exist_ok=False);wdc_evaluate(config['wdc'],out)
    else:
        path=ROOT/'results/latest.json'
        print(path.read_text(encoding='utf-8') if path.exists() else 'No completed research run')

if __name__=='__main__':main()
