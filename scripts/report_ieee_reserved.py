"""Read-only audit and paper exports for the completed one-time reserved run.

No inference, training, calibration or policy selection. Transaction-level data
are read locally for auditing and never exported to aggregate paper artifacts.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ieee_multiseed import ROOT, SEEDS, METHODS, read, sha, write, verify_hashes
from ieee_final_data import load_partition
from ieee_final_evaluation import METRICS, verify_contract, verify_validation_receipt
from report_ieee_validation import independent_metrics, summarize_rows, save_csv, LABELS, TITLES


def verify_reserved_receipt(directory, contract, validation_digest):
    directory=Path(directory)
    receipt=read(directory/'receipt.json')
    prepared=contract['preparation']
    if (receipt['status']!='complete' or receipt['phase']!='reserved'
            or receipt['contract_sha256']!=contract['contract_sha256']
            or receipt.get('validation_receipt_sha256')!=validation_digest
            or receipt.get('updates_after_development')!=0
            or not receipt['all_predictions_saved_before_metrics']):
        raise ValueError('Reserved completion/validation lineage mismatch')
    data=receipt['data']
    if (data['phase']!='reserved' or data['rows']!=prepared['partitions']['reserved_evaluation']['row_count']
            or data['consumed_rows']!=prepared['source_row_count'] or data['full_source_verified'] is not True
            or data['consumed_source_sha256']!=prepared['source_sha256']):
        raise ValueError('Reserved source/count verification mismatch')
    expected={f'predictions-{s}-{m}.npz' for s in SEEDS for m in METHODS}|{'metrics.json'}
    if set(receipt['files_sha256'])!=expected:
        raise ValueError('Incomplete reserved prediction set')
    verify_hashes({str(directory/name):value for name,value in receipt['files_sha256'].items()})
    if read(directory/'status.json')['status']!='complete':
        raise ValueError('Incomplete reserved status')
    marker=read(directory/'predictions-complete.json')
    predictions={k:v for k,v in receipt['files_sha256'].items() if k!='metrics.json'}
    if marker['files_sha256']!=predictions or marker['contract_sha256']!=contract['contract_sha256']:
        raise ValueError('Prediction completion marker mismatch')
    timestamp=(directory/'metrics.json').stat().st_mtime_ns
    if (directory/'predictions-complete.json').stat().st_mtime_ns>timestamp or any(
            (directory/name).stat().st_mtime_ns>timestamp for name in predictions):
        raise ValueError('Predictions not persisted before metrics')
    return receipt


def audit(contract_path, validation_path, reserved_path, prior_manifest):
    contract=read(contract_path);verify_contract(contract)
    validation=verify_validation_receipt(validation_path,contract)
    directory=Path(reserved_path).resolve()
    receipt=verify_reserved_receipt(directory,contract,sha(validation_path))
    frame,lineage=load_partition(contract['source'],contract['preparation'],'reserved',
                                 expected_prefix=validation['data']['validation_prefix_sha256'])
    if lineage!=receipt['data']:
        raise ValueError('Reserved raw source lineage differs from execution')
    parts=contract['preparation']['partitions']
    if (frame.TransactionDT.min()<=parts['validation']['max_transaction_dt']
            or frame.TransactionDT.min()<=contract['training_cutoff'] or not frame.TransactionID.is_unique):
        raise ValueError('Reserved chronological/ID separation failed')
    reserved_ids=set(frame.TransactionID.astype(int))
    # All validation arrays are already hash-verified; one supplies membership.
    with np.load(Path(validation_path).parent/f'predictions-{SEEDS[0]}-{METHODS[0]}.npz',allow_pickle=False) as data:
        if reserved_ids & set(data['transaction_id'].tolist()):
            raise ValueError('Reserved/validation ID overlap')
    for path in contract['artifacts_sha256']:
        if path.endswith('-metadata.csv'):
            if reserved_ids & set(pd.read_csv(path,usecols=['TransactionID']).TransactionID.astype(int)):
                raise ValueError('Reserved/development ID overlap')
    saved=read(directory/'metrics.json')
    if saved['phase']!='reserved' or saved['contract_sha256']!=contract['contract_sha256']:
        raise ValueError('Wrong metrics phase/contract')
    rows,checks=[],[]
    for record in contract['models']:
        name=f'predictions-{record["seed"]}-{record["method"]}.npz'
        with np.load(directory/name,allow_pickle=False) as data:
            if set(data.files)!={'transaction_id','scores'}:
                raise ValueError('Wrong prediction schema')
            ids,scores=data['transaction_id'],data['scores']
        if (ids.shape!=(len(frame),) or scores.shape!=(len(frame),)
                or not np.array_equal(ids,frame.TransactionID.to_numpy())):
            raise ValueError('Reserved prediction membership mismatch')
        result=independent_metrics(scores,frame)
        expected=next(r for r in saved['per_seed'] if (r['seed'],r['method'])==(record['seed'],record['method']))
        for key,value in result.items():
            if value is None:
                if expected[key] is not None:raise ValueError('Undefined metric mismatch')
            elif not math.isclose(value,expected[key],abs_tol=1e-12,rel_tol=1e-12):
                raise ValueError(f'Independent metric mismatch: {record["seed"]}/{record["method"]}/{key}')
        rows.append({'seed':record['seed'],'method':record['method'],**result})
        checks.append({'seed':record['seed'],'method':record['method'],'rows':len(frame),
                       'quota':result['review_quota_rows'],'membership_finite_scores_and_metrics_pass':True})
    summary,paired,contrasts=summarize_rows(rows)
    for actual,expected in zip(summary,saved['summary']):
        if actual['method']!=expected['method']:raise ValueError('Summary method mismatch')
        for metric in METRICS:
            for stat in ('mean','sd'):
                a,b=actual[metric][stat],expected[metric][stat]
                if (a is None)!=(b is None) or (a is not None and not math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12)):
                    raise ValueError('Aggregate mismatch')
    for metric in METRICS:
        if contrasts[metric]['available']!=saved['contrasts'][metric]['available']:
            raise ValueError('Contrast availability mismatch')
        if contrasts[metric]['available']:
            for key in ('mean','sd','positive','negative','ties'):
                if not math.isclose(contrasts[metric][key],saved['contrasts'][metric][key],abs_tol=1e-12):
                    raise ValueError('Paired contrast mismatch')
            for a,b in zip(contrasts[metric]['paired_t_95_interval'],saved['contrasts'][metric]['paired_t_95_interval']):
                if not math.isclose(a,b,abs_tol=1e-12):raise ValueError('Paired interval mismatch')
    verify_hashes(read(prior_manifest));verify_contract(contract)
    return {'phase':'reserved','starting_commit':'9de6406','rows':len(frame),'fraud_count':int(frame.isFraud.sum()),
            'review_quota_rows':math.ceil(.05*len(frame)),
            'reserved_time_range':[int(frame.TransactionDT.min()),int(frame.TransactionDT.max())],
            'source_lineage':lineage,'contract_sha256':contract['contract_sha256'],
            'contract_file_sha256':sha(contract_path),'reserved_receipt_sha256':sha(directory/'receipt.json'),
            'validation_receipt_sha256':sha(validation_path),'summary':summary,'per_seed':rows,
            'paired_differences':paired,'contrasts':contrasts,'prediction_audits':checks,
            'verification':{'all_25_saved_prediction_hashes_match':True,'all_25_independent_metrics_match':True,
                'full_source_sha256_matches_preparation':True,'validation_prefix_unchanged':True,
                'models_preprocessing_settings_protocol_unchanged':True,'no_training_or_tuning':True,
                'all_prior_artifacts_unchanged':True,'partition_ids_disjoint':True,'strict_time_separation':True,
                'prediction_files_precede_metrics':True,'quota_consistent':True},
            'report_source_sha256':sha(Path(__file__))}


def export(report,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'verification-and-results.json',report)
    flat=[{'method':r['method'],**{f'{m}_{s}':r[m][s] for m in METRICS for s in ('mean','sd')}} for r in report['summary']]
    save_csv(output/'method-summary.csv',flat);save_csv(output/'per-seed.csv',report['per_seed'])
    save_csv(output/'paired-differences.csv',report['paired_differences'])
    lines=[f'Reserved evaluation: {report["rows"]:,} transactions. Mean ± sample SD over seeds 501–505; fixed 5% review quota.', '',
           '| Method | AP | Fraud recall @5% | Amount-weighted fraud recall @5% |','| --- | --- | --- | --- |']
    for r in report['summary']:
        lines.append('| '+r['method']+' | '+' | '.join(f'{r[m]["mean"]:.6f} ± {r[m]["sd"]:.6f}' for m in METRICS)+' |')
    lines+=['','Paired differences: class normalization minus original combined.','',
            '| Seed | AP difference | Fraud recall difference | Amount recall difference |','| --- | --- | --- | --- |']
    for r in report['paired_differences']:
        lines.append('| '+str(r['seed'])+' | '+' | '.join(f'{r[m]:+.6f}' for m in METRICS)+' |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print('Contrasts:',report['contrasts'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for metric,ax in zip(METRICS,axes):
        for i,r in enumerate(report['summary']):
            values=[p[metric] for p in report['per_seed'] if p['method']==r['method']]
            ax.scatter(i+np.linspace(-.12,.12,5),values,s=20,color='#5892b9')
            ax.errorbar(i,r[metric]['mean'],yerr=r[metric]['sd'],fmt='o',capsize=4,color='#182e41')
        ax.set(title=TITLES[metric],xticks=range(5),xticklabels=[LABELS[m] for m in METHODS])
        ax.tick_params(axis='x',rotation=30,labelsize=9);ax.grid(axis='y',alpha=.25)
    fig.suptitle('One-time reserved evaluation: points = seeds; mean ± sample SD');fig.tight_layout()
    save_figure(fig,output/'reserved-comparison');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for metric,ax in zip(METRICS,axes):
        values=[r[metric]*100 for r in report['paired_differences']]
        mean=report['contrasts'][metric]['mean']*100
        ax.bar(range(5),values,color=['#337ca0' if v>=0 else '#ce6a35' for v in values])
        ax.axhline(0,color='#333',linewidth=.8);ax.axhline(mean,color='#555',linestyle='--',label=f'Mean {mean:+.3f}')
        ax.set(title=TITLES[metric],xticks=range(5),xticklabels=SEEDS,xlabel='Seed',ylabel='Class norm − combined (×100)')
        ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save_figure(fig,output/'paired-reserved');plt.close(fig)
    # Validation and reserved share frozen models/metrics; development windows
    # have different aggregation and are discussed separately in the document.
    validation=read(ROOT/'summaries/ieee-validation-v1/verification-and-results.json')
    cross=[{'partition':phase,'method':r['method'],**{f'{m}_{s}':r[m][s] for m in METRICS for s in ('mean','sd')}}
           for phase,result in [('validation',validation),('reserved',report)] for r in result['summary']]
    save_csv(output/'validation-reserved.csv',cross)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for metric,ax in zip(METRICS,axes):
        for i,method in enumerate(METHODS):
            a=next(r for r in validation['summary'] if r['method']==method)
            b=next(r for r in report['summary'] if r['method']==method)
            ax.plot([0,1],[a[metric]['mean'],b[metric]['mean']],marker='o',label=LABELS[method])
        ax.set(title=TITLES[metric],xticks=[0,1],xticklabels=['Validation','Reserved']);ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.tight_layout();save_figure(fig,output/'validation-reserved');plt.close(fig)


def save_figure(fig,path):
    for extension in ('png','pdf'):
        fig.savefig(f'{path}.{extension}',dpi=170,
                    metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True,help='Completed validation receipt')
    parser.add_argument('--reserved',type=Path,required=True,help='Completed reserved output directory')
    parser.add_argument('--prior-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    export(audit(args.contract,args.validation,args.reserved,args.prior_manifest),args.output)


if __name__=='__main__':main()
