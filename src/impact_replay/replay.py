"""Bounded uniform reservoir admission; interchangeable, class-guarded retrieval."""
import numpy as np

POLICIES=('random','forgetting','impact','combined')

def normalize_positive(x):
    x=np.maximum(np.asarray(x,dtype=float),0)
    scale=np.quantile(x,.95) if len(x) else 0
    return np.clip(x/scale,0,1) if scale>0 else np.zeros_like(x)

def score(loss_increase, impact, policy, alpha=.5, combination='sum'):
    if policy not in POLICIES:raise ValueError(policy)
    if not 0<=alpha<=1:raise ValueError('alpha must be in [0,1]')
    if combination not in ('sum','product'):raise ValueError(combination)
    f=normalize_positive(loss_increase);v=np.asarray(impact)
    if policy=='forgetting':return f
    if policy=='impact':return v
    if policy=='random':return np.zeros_like(f)
    if combination=='sum':return alpha*f+(1-alpha)*v
    return (f+1e-6)**alpha*(v+1e-6)**(1-alpha)

class Reservoir:
    def __init__(self,capacity,seed):
        if capacity<0:raise ValueError('negative capacity')
        self.capacity=capacity;self.rng=np.random.default_rng(seed);self.seen=0;self.rows=[]
    def add(self,rows):
        for row in rows:
            self.seen+=1
            if len(self.rows)<self.capacity:self.rows.append(row)
            else:
                j=int(self.rng.integers(self.seen))
                if j<self.capacity:self.rows[j]=row
    def select(self,budget,losses,policy,rng,alpha=.5,combination='sum',fraud_fraction=.25):
        if budget<0 or not 0<=fraud_fraction<=1:raise ValueError('invalid budget/quota')
        k=min(budget,len(self.rows))
        if k==0:return np.array([],dtype=int)
        y=np.array([r['y'] for r in self.rows]);ref=np.array([r['reference_loss'] for r in self.rows])
        impact=np.array([r['impact'] for r in self.rows])*y
        scores=score(np.asarray(losses)-ref,impact,policy,alpha,combination)
        def ranked(indices):
            order=rng.permutation(indices)
            return order[np.argsort(-scores[order],kind='stable')]
        pos=ranked(np.flatnonzero(y==1));neg=ranked(np.flatnonzero(y==0))
        kp=min(int(round(k*fraud_fraction)),len(pos));kn=min(k-kp,len(neg));kp=min(k-kn,len(pos))
        chosen=np.concatenate([pos[:kp],neg[:kn]])
        assert len(chosen)==k and len(set(chosen))==k
        return chosen

class StratifiedReservoir(Reservoir):
    """Two true reservoirs; total capacity fixed, common admission across policies."""
    def __init__(self,capacity,seed,fraud_fraction=.5):
        if capacity<0 or not 0<=fraud_fraction<=1:raise ValueError('invalid memory quota')
        self.capacity=capacity
        self.positive=Reservoir(int(round(capacity*fraud_fraction)),seed)
        self.negative=Reservoir(capacity-self.positive.capacity,seed+100)
    @property
    def rows(self):return self.positive.rows+self.negative.rows
    @property
    def seen(self):return self.positive.seen+self.negative.seen
    def add(self,rows):
        self.positive.add([r for r in rows if r['y']==1]);self.negative.add([r for r in rows if r['y']==0])
    def state(self):
        return {name:{'seen':r.seen,'rng':r.rng.bit_generator.state,'capacity':r.capacity} for name,r in [('positive',self.positive),('negative',self.negative)]}
