import argparse, json, random, time, hashlib, sys, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from .replay import Reservoir,StratifiedReservoir
from .metrics import metrics,forgetting
from .storage import save_npz,save_checkpoint

def run(config,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for name in ['experiment.py','replay.py','metrics.py','storage.py']:
        (out/name).write_text((Path(__file__).parent/name).read_text())
    (out/'config.json').write_text(json.dumps(config,indent=2))
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    shuffle=np.random.default_rng(seed+1); retrieval=np.random.default_rng(seed+2)
    data_path=Path(config['data']);d=pd.read_csv(data_path).sort_values(['TX_TIME_SECONDS','TRANSACTION_ID']).reset_index(drop=True)
    required=['TRANSACTION_ID','TX_TIME_SECONDS','TX_TIME_DAYS','TX_AMOUNT','TX_FRAUD','TX_FRAUD_SCENARIO']
    if d[required].isna().any().any() or d.TRANSACTION_ID.duplicated().any():raise ValueError('Invalid data')
    if not d.TX_FRAUD.isin([0,1]).all() or (d.TX_AMOUNT<0).any():raise ValueError('Invalid label/amount')
    # Only contemporaneously available values. IDs and scenario labels never enter model.
    hour=(d.TX_TIME_SECONDS%86400)/3600
    raw=np.column_stack([np.log1p(d.TX_AMOUNT),np.sin(hour*2*np.pi/24),np.cos(hour*2*np.pi/24),(d.TX_TIME_DAYS%7>=5).astype(float)])
    y=d.TX_FRAUD.to_numpy(dtype=int);amount=d.TX_AMOUNT.to_numpy();ids=d.TRANSACTION_ID.to_numpy();scenario=d.TX_FRAUD_SCENARIO.to_numpy()
    windows=(d.TX_TIME_DAYS//config['window_days']).to_numpy(dtype=int)
    splits=[]
    for w in sorted(set(windows)):
        idx=np.flatnonzero(windows==w);n=len(idx);a=int(.6*n);b=int(.8*n)
        if not (0<a<b<n):raise ValueError('Window too small')
        splits.append((idx[:a],idx[a:b],idx[b:]))
    warm=splits[0][0];scaler=StandardScaler().fit(raw[warm]);x=torch.tensor(scaler.transform(raw),dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32)
    impact_scale=max(float(np.quantile(np.log1p(amount[warm]),.95)),1e-6)
    impact=np.clip(np.log1p(amount)/impact_scale,0,1)
    model=nn.Sequential(nn.Linear(x.shape[1],config['hidden']),nn.ReLU(),nn.Linear(config['hidden'],1))
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    weight=float((1-y[warm]).sum()/max(1,y[warm].sum())) if config['class_weight'] else 1.
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    memory=(StratifiedReservoir(config['memory'],seed+3,config.get('memory_fraud_fraction',.5)) if config.get('memory_admission')=='stratified' else Reservoir(config['memory'],seed+3))
    threshold=0.5
    def probs(indices):
        model.eval()
        with torch.no_grad():return torch.sigmoid(model(x[indices]).flatten()).numpy()
    def memory_losses():
        if not memory.rows:return np.array([])
        model.eval()
        with torch.no_grad():
            xx=torch.tensor(np.array([r['x'] for r in memory.rows]),dtype=torch.float32)
            yy=torch.tensor([r['y'] for r in memory.rows],dtype=torch.float32)
            return nn.functional.binary_cross_entropy_with_logits(model(xx).flatten(),yy,reduction='none').numpy()
    history=[];matrix=np.full((len(splits),len(splits)),np.nan);amount_matrix=matrix.copy()
    manifest={'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),'features':['log_amount','hour_sin','hour_cos','weekend'],'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'impact_scale':impact_scale,'positive_weight':weight,'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__},'protocol':'60/20/20 chronological within each window; validation used only for configured threshold calibration; test never trained or replayed; immediate labels for training prefix only'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save_npz(out/'splits.npz',**{f'{k}_{i}':ids[z] for i,split in enumerate(splits) for k,z in zip(['train','validation','test'],split)})
    start=time.time();all_test=set(np.concatenate([ids[s[2]] for s in splits]).tolist());selected_log=[]
    for t,(train,val,test) in enumerate(splits):
        pre=metrics(y[test],probs(test),amount[test],threshold); nreplay=0;steps=0
        for epoch in range(config['epochs']):
            # One capped historical draw per epoch, distributed over current minibatches.
            chosen=memory.select(config['replay_budget'],memory_losses(),config['policy'],retrieval,config['alpha'],config['combination'],config['fraud_fraction']) if config['policy']!='none' else np.array([],dtype=int)
            selected=[memory.rows[i] for i in chosen];nreplay+=len(selected)
            if selected:
                assert not all_test.intersection(r['id'] for r in selected)
                assert all(r['window']<t for r in selected)
            selected_log.append({'window':t,'epoch':epoch,'ids':[r['id'] for r in selected]})
            order=shuffle.permutation(train);batches=[order[i:i+config['batch_size']] for i in range(0,len(order),config['batch_size'])]
            replay_parts=np.array_split(np.arange(len(selected)),len(batches))
            for current,part in zip(batches,replay_parts):
                xx=x[current];yy=yt[current]
                if len(part):
                    xx=torch.cat([xx,torch.tensor(np.array([selected[i]['x'] for i in part]),dtype=torch.float32)])
                    yy=torch.cat([yy,torch.tensor([selected[i]['y'] for i in part],dtype=torch.float32)])
                model.train();optimizer.zero_grad();loss=criterion(model(xx).flatten(),yy)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                loss.backward();optimizer.step();steps+=1
        if config.get('threshold_mode')=='current_validation_fpr' or (t==0 and config.get('threshold_mode')=='warmup_validation_fpr'):
            vp=probs(val); legitimate=vp[y[val]==0]
            if not len(legitimate):raise ValueError('No legitimate validation data')
            threshold=float(np.quantile(legitimate,1-config['target_fpr'],method='higher'))
        model.eval()
        with torch.no_grad():ref=nn.functional.binary_cross_entropy_with_logits(model(x[train]).flatten(),yt[train],reduction='none').numpy()
        memory.add([dict(id=int(ids[i]),x=x[i].tolist(),y=int(y[i]),amount=float(amount[i]),impact=float(impact[i]),reference_loss=float(l),window=t) for i,l in zip(train,ref)])
        assert len(memory.rows)<=config['memory']
        historical=[]
        for j in range(t+1):
            idx=splits[j][2];p=probs(idx);m=metrics(y[idx],p,amount[idx],threshold);historical.append(m);matrix[j,t]=m['ap'] if m['ap'] is not None else np.nan;amount_matrix[j,t]=m['amount_recall'] if m['amount_recall'] is not None else np.nan
        # Purely diagnostic, never used to tune selection or thresholds.
        p=probs(test);sub=scenario[test]!=1
        row={'window':t,'prequential':pre,'post_update':historical[-1],'historical':historical,'excluding_scenario1':metrics(y[test][sub],p[sub],amount[test][sub],threshold),'memory_size':len(memory.rows),'memory_frauds':sum(r['y'] for r in memory.rows),'replayed_total':nreplay,'optimizer_steps':steps,'train_n':len(train)}
        history.append(row)
        save_checkpoint({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'memory':memory.rows,'seen':memory.seen,'reservoir_state':memory.state() if isinstance(memory,StratifiedReservoir) else {'seen':memory.seen,'rng':memory.rng.bit_generator.state},'shuffle_rng':shuffle.bit_generator.state,'retrieval_rng':retrieval.bit_generator.state,'torch_rng':torch.get_rng_state(),'config':config,'manifest':manifest,'window':t,'threshold':threshold},out/f'checkpoint-{t:02d}.pt')
    (out/'metrics.jsonl').write_text(''.join(json.dumps(item,allow_nan=False)+'\n' for item in history))
    np.save(out/'ap_matrix.npy',matrix);np.save(out/'amount_recall_matrix.npy',amount_matrix)
    (out/'replay_ids.json').write_text(json.dumps(selected_log))
    last=history[-1]['historical'];summary={'seed':seed,'policy':config['policy'],'windows':len(splits),'seconds':time.time()-start,'mean_prequential_ap':float(np.mean([r['prequential']['ap'] for r in history[1:] if r['prequential']['ap'] is not None])),'final_historical_ap':float(np.nanmean(matrix[:,-1])),'final_historical_amount_recall':float(np.nanmean(amount_matrix[:,-1])),'ap_forgetting':forgetting(matrix),'amount_recall_forgetting':forgetting(amount_matrix),'final_historical_missed_amount':sum(r['missed_fraud_amount'] for r in last),'final_historical_false_positives':sum(r['false_positives'] for r in last),'final_precision_at_1pct':float(np.mean([r['precision_at_1pct'] for r in last])), 'final_amount_recall_at_1pct':float(np.mean([r['amount_recall_at_1pct'] for r in last if r['amount_recall_at_1pct'] is not None])),'threshold':threshold,'status':'engineering_pilot_not_confirmatory'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--policy');p.add_argument('--seed',type=int);a=p.parse_args();c=json.loads(Path(a.config).read_text())
    if a.policy:c['policy']=a.policy
    if a.seed is not None:c['seed']=a.seed
    run(c,a.out)
if __name__=='__main__':main()
