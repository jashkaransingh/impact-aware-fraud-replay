import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('controlled');policies=['none','random','forgetting','impact','combined'];labels=['No replay','Random','Forgetting','Impact','Combined'];colors=['#9d3b3b','#64748b','#237b53','#b48426','#4d53a3']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(13,5.4));ax[0].axvspan(2.5,5.5,color='#edf0f5',zorder=0)
for policy,label,color in zip(policies,labels,colors):
 arrays=[]
 for seed in range(201,206):
  h=json.loads((r/f'test-independent-{policy}-{seed}'/'history.json').read_text());arrays.append([z['after'][0]['ap'] for z in h])
 a=np.array(arrays);mean=a.mean(0);sd=a.std(0,ddof=1);ax[0].plot(range(1,10),mean,label=label,color=color,lw=2);ax[0].fill_between(range(1,10),np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=color,alpha=.08)
ax[0].set(title='A retention through A → B → A',xlabel='Training window',ylabel='Average precision on fixed A test set',ylim=(0,1.08),xticks=range(1,10));ax[0].text(2,1.035,'A',ha='center');ax[0].text(5,1.035,'B',ha='center');ax[0].text(8,1.035,'A returns',ha='center');ax[0].legend(loc='lower right',ncol=2,fontsize=9)
for j,(file,label,color) in enumerate([('test-summary.csv','Memory 256 / replay 128','#4d53a3'),('tight-summary.csv','Memory 64 / replay 32','#b48426')]):
 d=pd.read_csv(r/file);p=d.pivot(index=['amount_mode','seed'],columns='policy',values='pooled_amount_recall_final');delta=100*(p['combined']-p['forgetting']);groups=delta.groupby(level=0);order=['independent','a_high','b_high'];means=groups.mean().reindex(order);sd=groups.std().reindex(order);x=np.arange(3)+(j-.5)*.32
 ax[1].bar(x,means,width=.3,color=color,label=label,yerr=sd,capsize=4,alpha=.9)
ax[1].axhline(0,color='#333',lw=1);ax[1].set(title='Adding impact: gains depend on the setting',ylabel='Combined − forgetting-only\nfinal amount recall (percentage points)',xticks=np.arange(3),xticklabels=['Independent\namounts','A amounts\n10× for fraud','B amounts\n10× for fraud']);ax[1].legend(fontsize=9)
fig.suptitle('Controlled synthetic stress test — preliminary evidence',fontsize=15,y=.99)
fig.text(.5,.015,'Means and ±1 SD over five seeds per budget. Amount never determines the label. Right: fixed pooled 5% review budget.',ha='center',fontsize=9,color='#444')
fig.tight_layout(rect=(0,.065,1,.94));fig.savefig(r/'controlled-results.png',dpi=160);plt.close(fig)
