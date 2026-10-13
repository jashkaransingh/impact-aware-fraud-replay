import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

def metrics(y, p, amount, threshold=0.5):
    y=np.asarray(y,dtype=int); p=np.asarray(p); amount=np.asarray(amount)
    pred=p>=threshold; positive=y==1; missed=positive & ~pred
    total=float(amount[positive].sum())
    result=dict(n=len(y),frauds=int(y.sum()),ap=float(average_precision_score(y,p)) if y.sum() else None,
        auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)),false_positives=int((pred & ~positive).sum()),
        amount_recall=float(amount[positive & pred].sum()/total) if total else None,
        missed_fraud_amount=float(amount[missed].sum()),threshold=threshold)
    k=max(1,int(np.ceil(.01*len(y)))) if len(y) else 0
    selected=np.argsort(-p,kind='stable')[:k]
    result.update(review_budget_1pct=k,precision_at_1pct=float(y[selected].mean()) if k else None,
        amount_recall_at_1pct=float((amount[selected]*y[selected]).sum()/total) if total else None)
    return result

def forgetting(matrix):
    """Final-column drop from best prior performance, excluding newest task."""
    m=np.array(matrix,dtype=float); t=m.shape[1]-1
    if t<1:return None
    drops=[np.nanmax(m[j,j:t])-m[j,t] for j in range(t) if np.isfinite(m[j,t]) and np.isfinite(m[j,j:t]).any()]
    return float(np.mean(drops)) if drops else None
