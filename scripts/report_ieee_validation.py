"""Audit saved validation predictions and export aggregate paper artifacts.

Never trains or predicts. Reads only the development+validation CSV prefix;
reserved access is not supported. Transaction-level arrays stay in results/.
"""
import argparse
import csv
import math
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import average_precision_score

from ieee_multiseed import ROOT, SEEDS, METHODS, read, sha, write, verify_hashes
from ieee_final_data import load_partition
from ieee_final_evaluation import METRICS, verify_contract, verify_validation_receipt

LABELS = {'no-replay': 'No replay', 'random': 'Random', 'forgetting': 'Forgetting',
          'combined': 'Combined', 'combined-classnorm': 'Class norm'}
TITLES = {'ap': 'Average precision', 'fraud_recall_at_5pct': 'Fraud recall at 5%',
          'amount_weighted_fraud_recall_at_5pct': 'Fraud amount recall at 5%'}


def independent_metrics(scores, frame):
    """Separate ranking implementation; official sklearn AP, same fixed quota."""
    if len(scores) != len(frame) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid prediction array')
    ranking = pd.DataFrame({'score': scores, 'id': frame.TransactionID.to_numpy(),
                            'label': frame.isFraud.to_numpy(), 'amount': frame.TransactionAmt.to_numpy()})
    top = ranking.sort_values(['score','id'], ascending=[False,True], kind='mergesort').iloc[:math.ceil(.05*len(frame))]
    positives = int(ranking.label.sum())
    total = math.fsum(ranking.loc[ranking.label == 1, 'amount'])
    return {'rows': len(frame), 'fraud_count': positives, 'review_quota_rows': len(top),
            'ap': float(average_precision_score(ranking.label,scores)) if positives else None,
            'fraud_recall_at_5pct': float(top.label.sum()/positives) if positives else None,
            'amount_weighted_fraud_recall_at_5pct': math.fsum(top.loc[top.label == 1,'amount'])/total if total else None}


def summarize_rows(rows):
    if len(rows) != 25 or {(r['seed'],r['method']) for r in rows} != {(s,m) for s in SEEDS for m in METHODS}:
        raise ValueError('Expected all 25 method/seed pairs')
    summary = []
    for method in METHODS:
        entry = {'method': method}
        for metric in METRICS:
            values = [r[metric] for r in rows if r['method'] == method]
            if any(v is None for v in values):
                entry[metric] = {'mean':None,'sd':None}
            else:
                entry[metric] = {'mean':statistics.mean(values),'sd':statistics.stdev(values)}
        summary.append(entry)
    lookup = {(r['seed'],r['method']): r for r in rows}
    paired = [{'seed':s, **{m:lookup[s,'combined-classnorm'][m]-lookup[s,'combined'][m]
                 if lookup[s,'combined-classnorm'][m] is not None and lookup[s,'combined'][m] is not None else None
                 for m in METRICS}} for s in SEEDS]
    contrasts = {}
    for metric in METRICS:
        values = [r[metric] for r in paired]
        if any(v is None for v in values):
            contrasts[metric] = {'available':False}; continue
        mean, sd = statistics.mean(values), statistics.stdev(values)
        halfwidth = float(t.ppf(.975,4))*sd/math.sqrt(5)
        contrasts[metric] = {'available':True, 'mean':mean,'sd':sd,
            'paired_t_95_interval':[mean-halfwidth,mean+halfwidth],
            'positive':sum(v>0 for v in values),'negative':sum(v<0 for v in values),'ties':sum(v==0 for v in values)}
    return summary, paired, contrasts


def audit(contract_path, validation_path):
    contract, directory = read(contract_path), Path(validation_path).resolve()
    verify_contract(contract)
    receipt = verify_validation_receipt(directory/'receipt.json',contract)
    if read(directory/'status.json')['status'] != 'complete':
        raise ValueError('Validation incomplete')
    marker = read(directory/'predictions-complete.json')
    expected_predictions = {k:v for k,v in receipt['files_sha256'].items() if k != 'metrics.json'}
    if marker['files_sha256'] != expected_predictions or marker['contract_sha256'] != contract['contract_sha256']:
        raise ValueError('Prediction completion marker mismatch')
    metric_time = (directory/'metrics.json').stat().st_mtime_ns
    if (directory/'predictions-complete.json').stat().st_mtime_ns > metric_time or any(
            (directory/name).stat().st_mtime_ns > metric_time for name in expected_predictions):
        raise ValueError('Predictions not persisted before metric file')
    frame, lineage = load_partition(contract['source'],contract['preparation'],'validation')
    if lineage != receipt['data'] or lineage['full_source_verified']:
        raise ValueError('Consumed validation source prefix changed')
    parts = contract['preparation']['partitions']
    if (len(frame) != parts['validation']['row_count'] or frame.TransactionDT.min() <= contract['training_cutoff']
            or frame.TransactionDT.max() >= parts['reserved_evaluation']['min_transaction_dt']
            or not frame.TransactionID.is_unique):
        raise ValueError('Validation temporal/ID separation failed')
    if 'fraud_count' in parts['validation'] and int(frame.isFraud.sum()) != parts['validation']['fraud_count']:
        raise ValueError('Validation fraud count differs from preparation')
    # Independent membership check against every prepared development subset.
    validation_ids = set(frame.TransactionID.astype(int))
    metadata_paths = [p for p in contract['artifacts_sha256'] if p.endswith('-metadata.csv')]
    if len(metadata_paths) != 12:
        raise ValueError('Expected 12 development metadata artifacts')
    for path in metadata_paths:
        ids = set(pd.read_csv(path,usecols=['TransactionID']).TransactionID.astype(int))
        if ids & validation_ids:
            raise ValueError('Validation ID occurs in development data')
    saved = read(directory/'metrics.json')
    rows, checks = [], []
    for record in contract['models']:
        name = f'predictions-{record["seed"]}-{record["method"]}.npz'
        with np.load(directory/name,allow_pickle=False) as data:
            if set(data.files) != {'transaction_id','scores'}:
                raise ValueError('Wrong prediction schema')
            ids,scores = data['transaction_id'],data['scores']
        if ids.shape != (len(frame),) or scores.shape != (len(frame),) or not np.array_equal(ids,frame.TransactionID.to_numpy()):
            raise ValueError('Prediction membership/order mismatch')
        metric = independent_metrics(scores,frame)
        original = next(r for r in saved['per_seed'] if (r['seed'],r['method']) == (record['seed'],record['method']))
        for key,value in metric.items():
            if value is None:
                if original[key] is not None: raise ValueError('Undefined metric mismatch')
            elif not math.isclose(value,original[key],rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError(f'Independent metric mismatch: {record["seed"]}/{record["method"]}/{key}')
        rows.append({'seed':record['seed'],'method':record['method'],**metric})
        checks.append({'seed':record['seed'],'method':record['method'],'rows':len(scores),
                       'review_quota_rows':metric['review_quota_rows'],'finite_scores':True,
                       'same_validation_membership':True,'independent_metrics_match':True})
    summary,paired,contrasts = summarize_rows(rows)
    for expected,actual in zip(saved['summary'],summary):
        for metric in METRICS:
            for stat in ('mean','sd'):
                if not math.isclose(expected[metric][stat],actual[metric][stat],rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Aggregate metric mismatch')
    registry = read(ROOT/'results/ieee-multiseed-501-505/registry.json')
    verify_hashes(registry['protected_existing_artifacts'])
    verify_hashes(registry['prepared_data']['files_sha256'])
    verify_contract(contract)
    return {'phase':'validation','contract_sha256':contract['contract_sha256'],
            'contract_file_sha256':sha(contract_path),'validation_receipt_sha256':sha(directory/'receipt.json'),
            'rows':len(frame),'fraud_count':int(frame.isFraud.sum()),'review_quota_rows':math.ceil(.05*len(frame)),
            'validation_time_range':[int(frame.TransactionDT.min()),int(frame.TransactionDT.max())],
            'training_cutoff':contract['training_cutoff'],'source_lineage':lineage,
            'summary':summary,'per_seed':rows,'paired_differences':paired,'contrasts':contrasts,
            'prediction_audits':checks,
            'verification':{'all_25_prediction_hashes_match':True,'all_frozen_artifact_hashes_match':True,
                'preprocessing_and_model_checkpoints_unchanged':True,'independent_metrics_match':True,
                'prediction_files_precede_metrics':True,'development_validation_disjoint':True,
                'reserved_rows_not_requested_or_decoded':True,'prior_results_and_prepared_inputs_unchanged':True,
                'full_source_sha_deferred_to_authorized_reserved_stage':True},
            'reporting_source_sha256':sha(Path(__file__))}


def save_csv(path, rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def export(report, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'verification-and-results.json',report)
    flat=[{'method':r['method'],**{f'{m}_{s}':r[m][s] for m in METRICS for s in ('mean','sd')}} for r in report['summary']]
    save_csv(output/'method-summary.csv',flat)
    save_csv(output/'per-seed.csv',report['per_seed'])
    save_csv(output/'paired-differences.csv',report['paired_differences'])
    lines=['Validation: 105,088 rows; mean ± sample SD across seeds 501–505. Fixed 5% batch review quota.', '',
           '| Method | AP | Fraud recall @5% | Amount-weighted fraud recall @5% |', '| --- | --- | --- | --- |']
    for row in report['summary']:
        lines.append('| '+row['method']+' | '+' | '.join(f'{row[m]["mean"]:.6f} ± {row[m]["sd"]:.6f}' for m in METRICS)+' |')
    lines += ['', 'Paired differences: class-normalized minus original combined.', '',
              '| Seed | AP difference | Fraud recall difference | Amount recall difference |','| --- | --- | --- | --- |']
    for row in report['paired_differences']:
        lines.append('| '+str(row['seed'])+' | '+' | '.join(f'{row[m]:+.6f}' for m in METRICS)+' |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for metric,ax in zip(METRICS,axes):
        for i,row in enumerate(report['summary']):
            values=[r[metric] for r in report['per_seed'] if r['method']==row['method']]
            ax.scatter(i+np.linspace(-.12,.12,5),values,color='#5892b9',s=20)
            ax.errorbar(i,row[metric]['mean'],yerr=row[metric]['sd'],fmt='o',capsize=4,color='#182e41')
        ax.set(title=TITLES[metric],xticks=range(5),xticklabels=[LABELS[m] for m in METHODS])
        ax.tick_params(axis='x',rotation=30,labelsize=9);ax.grid(axis='y',alpha=.25)
    fig.suptitle('Frozen-model validation: points = seeds; mean ± sample SD')
    fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(output/f'validation-comparison.{extension}',dpi=170,
                    metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else None)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for metric,ax in zip(METRICS,axes):
        values=[r[metric]*100 for r in report['paired_differences']]
        ax.bar(range(5),values,color=['#337ca0' if v>=0 else '#ce6a35' for v in values])
        mean=report['contrasts'][metric]['mean']*100
        ax.axhline(0,color='#333',linewidth=.8)
        ax.axhline(mean,color='#555',linestyle='--',label=f'Mean {mean:+.3f}')
        ax.set(title=TITLES[metric],xticks=range(5),xticklabels=SEEDS,xlabel='Seed',ylabel='Class norm − combined (×100)')
        ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(output/f'paired-validation.{extension}',dpi=170,
                    metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else None)
    plt.close(fig)
    print('\n'.join(lines))
    print('Contrasts:',report['contrasts'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    export(audit(args.contract,args.validation),args.output)


if __name__=='__main__':main()
