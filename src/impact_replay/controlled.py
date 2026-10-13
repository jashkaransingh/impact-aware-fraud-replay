"""Synthetic context-switch mechanism test with fixed-label Gaussian mixtures."""
import argparse,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .replay import StratifiedReservoir
from .storage import atomic_bytes,save_checkpoint,save_npz
from .metrics import metrics

def write_json(p,obj):atomic_bytes(p,json.dumps(obj,indent=2,allow_nan=False).encode())

def make_block(n,domain,rng,c,id_start):
    y=np.zeros(n,dtype=np.float32);y[:int(round(n*c['fraud_fraction_data']))]=1;rng.shuffle(y)
    sign=1 if domain==0 else -1
    centers=np.column_stack([(2*y-1)*sign*c['feature_center'],np.full(n,sign*c['feature_center']),np.zeros((n,2))])
    x=rng.normal(0,c['feature_sd'],size=(n,4))+centers
    a=rng.lognormal(np.log(50),1.0,n)
    if c['amount_mode']=='a_high':a[(y==1)&(domain==0)]*=10
    elif c['amount_mode']=='b_high':a[(y==1)&(domain==1)]*=10
    elif c['amount_mode']!='independent':raise ValueError('amount mode')
    return {'x':x.astype(np.float32),'y':y,'amount':a,'ids':np.arange(id_start,id_start+n),'domain':domain}

def evaluate(model,block):
    model.eval()
    with torch.no_grad():p=torch.sigmoid(model(torch.from_numpy(block['x'])).flatten()).numpy()
    y=block['y'];a=block['amount'];m=metrics(y,p,a)
    k=int(np.ceil(.05*len(y)));rank=np.argsort(-p,kind='stable')[:k]
    m.update(recall_at_5pct=float(y[rank].sum()/y.sum()),amount_recall_at_5pct=float((y[rank]*a[rank]).sum()/(y*a).sum()))
    return m

def run(c,out):
    if c['amount_as_feature']:raise ValueError('This protocol excludes amount from model inputs')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    random.seed(c['seed']);torch.manual_seed(c['seed']);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    # Independent substreams: data, evaluation, model init, ordering and retrieval.
    data_rng=np.random.default_rng(c['seed']);eval_rng=np.random.default_rng(c['seed']+100000)
    order_rng=np.random.default_rng(c['seed']+200000);select_rng=np.random.default_rng(c['seed']+300000)
    blocks=[make_block(c['train_per_window'],d,data_rng,c,i*c['train_per_window']) for i,d in enumerate(c['schedule'])]
    tests=[make_block(c['eval_per_domain'],d,eval_rng,c,10**8+d*c['eval_per_domain']) for d in [0,1]]
    model=nn.Sequential(nn.Linear(4,c['hidden']),nn.ReLU(),nn.Linear(c['hidden'],1))
    opt=torch.optim.SGD(model.parameters(),lr=c['lr']);criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-c['fraud_fraction_data'])/c['fraud_fraction_data']))
    memory=StratifiedReservoir(c['memory'],c['seed']+400000,.5)
    # Smooth strictly increasing impact scale; avoids the pilot's clipping saturation.
    scale=max(float(np.median(np.log1p(blocks[0]['amount']))),1e-6)
    history=[];replay_log=[];start=time.time()
    write_json(out/'config.json',c)
    for name in ['controlled.py','replay.py','metrics.py','storage.py']:
        atomic_bytes(out/name,(Path(__file__).parent/name).read_bytes())
    save_npz(out/'data.npz',**{f'{kind}_{i}_{k}':v for kind,items in [('train',blocks),('test',tests)] for i,b in enumerate(items) for k,v in b.items() if isinstance(v,np.ndarray)})
    write_json(out/'provenance.json',{'data_sha256':hashlib.sha256((out/'data.npz').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((out/'controlled.py').read_bytes()).hexdigest(),'synthetic':True,'evaluation_kind':c['evaluation_kind'],'amount_enters_label_rule':False,'impact_scale':scale})
    for t,b in enumerate(blocks):
        before=[evaluate(model,z) for z in tests];xx=torch.tensor(b['x']);yy=torch.tensor(b['y']);used=0;stepcount=0
        for epoch in range(c['epochs']):
            model.eval()
            if memory.rows:
                with torch.no_grad():losses=nn.functional.binary_cross_entropy_with_logits(model(torch.tensor(np.array([r['x'] for r in memory.rows]),dtype=torch.float32)).flatten(),torch.tensor([r['y'] for r in memory.rows],dtype=torch.float32),reduction='none').numpy()
            else:losses=[]
            chosen=memory.select(c['replay_budget'],losses,c['policy'],select_rng,alpha=c['alpha'],fraud_fraction=c['fraud_fraction_replay']) if c['policy']!='none' else []
            selected=[memory.rows[i] for i in chosen];used+=len(selected)
            assert all(r['window']<t for r in selected)
            replay_log.append({'window':t,'epoch':epoch,'ids':[r['id'] for r in selected],'domains':[r['domain'] for r in selected]})
            order=order_rng.permutation(len(xx));batches=[order[i:i+c['batch_size']] for i in range(0,len(order),c['batch_size'])];parts=np.array_split(np.arange(len(selected)),len(batches))
            for idx,part in zip(batches,parts):
                x=xx[idx];y=yy[idx]
                if len(part):
                    x=torch.cat([x,torch.tensor(np.array([selected[i]['x'] for i in part]),dtype=torch.float32)]);y=torch.cat([y,torch.tensor([selected[i]['y'] for i in part],dtype=torch.float32)])
                model.train();opt.zero_grad();loss=criterion(model(x).flatten(),y);assert torch.isfinite(loss);loss.backward();opt.step();stepcount+=1
        model.eval()
        with torch.no_grad():ref=nn.functional.binary_cross_entropy_with_logits(model(xx).flatten(),yy,reduction='none').numpy()
        loga=np.log1p(b['amount']);impact=loga/(loga+scale)
        memory.add([{'x':b['x'][i].tolist(),'y':int(b['y'][i]),'id':int(b['ids'][i]),'window':t,'domain':b['domain'],'reference_loss':float(ref[i]),'impact':float(impact[i])} for i in range(len(xx))])
        row={'window':t,'domain':b['domain'],'before':before,'after':[evaluate(model,z) for z in tests],'replayed':used,'steps':stepcount,'memory':len(memory.rows)};history.append(row)
        save_checkpoint({'model':model.state_dict(),'optimizer':opt.state_dict(),'config':c,'window':t,'memory':memory.rows,'memory_state':memory.state(),'order_rng':order_rng.bit_generator.state,'select_rng':select_rng.bit_generator.state,'torch_rng':torch.get_rng_state()},out/f'checkpoint-{t}.pt')
    mixed={k:np.concatenate([z[k] for z in tests]) for k in ['x','y','amount','ids']}
    mixed_final=evaluate(model,mixed)
    # Recover the phase-B checkpoint to measure pooled old+current-domain utility.
    final_state={k:v.clone() for k,v in model.state_dict().items()}
    phase_b=torch.load(out/'checkpoint-5.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(phase_b['model']);mixed_phase_b=evaluate(model,mixed);model.load_state_dict(final_state)
    h=history
    summary={'seed':c['seed'],'policy':c['policy'],'amount_mode':c['amount_mode'],'A_learned_ap':h[2]['after'][0]['ap'],'A_after_B_ap':h[5]['after'][0]['ap'],'A_forgetting':h[2]['after'][0]['ap']-h[5]['after'][0]['ap'],'B_learned_ap':h[5]['after'][1]['ap'],'A_return_ap':h[8]['after'][0]['ap'],'B_after_A_ap':h[8]['after'][1]['ap'],'B_forgetting':h[5]['after'][1]['ap']-h[8]['after'][1]['ap'],'A_amount_recall_before':h[2]['after'][0]['amount_recall_at_5pct'],'A_amount_recall_after_B':h[5]['after'][0]['amount_recall_at_5pct'],'B_amount_recall_before':h[5]['after'][1]['amount_recall_at_5pct'],'B_amount_recall_after_A':h[8]['after'][1]['amount_recall_at_5pct'],'pooled_amount_recall_after_B':mixed_phase_b['amount_recall_at_5pct'],'pooled_amount_recall_final':mixed_final['amount_recall_at_5pct'],'pooled_ap_after_B':mixed_phase_b['ap'],'pooled_ap_final':mixed_final['ap'],'seconds':time.time()-start,'evaluation_kind':c['evaluation_kind']}
    write_json(out/'history.json',history);write_json(out/'replay.json',replay_log);write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)
    return summary

def joint_control(c,out):
    """Capacity diagnostic: balanced exposure to both domains, not budget-matched baseline."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(c['seed']);torch.set_num_threads(1)
    rng=np.random.default_rng(c['seed']);ev=np.random.default_rng(c['seed']+100000);order=np.random.default_rng(c['seed']+200000)
    blocks=[make_block(c['train_per_window'],d,rng,c,i*c['train_per_window']) for i,d in enumerate(c['schedule'])];tests=[make_block(c['eval_per_domain'],d,ev,c,10**8+d*c['eval_per_domain']) for d in [0,1]]
    x=torch.tensor(np.concatenate([z['x'] for z in blocks]));y=torch.tensor(np.concatenate([z['y'] for z in blocks]));model=nn.Sequential(nn.Linear(4,c['hidden']),nn.ReLU(),nn.Linear(c['hidden'],1));opt=torch.optim.SGD(model.parameters(),lr=c['lr']);criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(19.))
    for epoch in range(c['epochs']):
        indices=order.permutation(len(x))
        for i in range(0,len(x),c['batch_size']):
            idx=indices[i:i+c['batch_size']];opt.zero_grad();loss=criterion(model(x[idx]).flatten(),y[idx]);loss.backward();opt.step()
    result={'seed':c['seed'],'domain_metrics':[evaluate(model,z) for z in tests],'status':'unconstrained_joint_capacity_control'};write_json(out/'config.json',c);write_json(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int);p.add_argument('--policy');p.add_argument('--joint',action='store_true');a=p.parse_args();c=json.loads(Path(a.config).read_text())
    if a.seed is not None:c['seed']=a.seed
    if a.policy:c['policy']=a.policy
    (joint_control if a.joint else run)(c,a.out)
