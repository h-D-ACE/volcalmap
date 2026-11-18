"""Learn a small perceptual distance head on frozen audio features, not identity IDs."""
import argparse,json,copy,random
from pathlib import Path
from runtime import ROOT,write_json
from dataset import load_data,make_input,pca_transform,unit
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr,pearsonr

class MetricHead(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__();self.projection=nn.Linear(input_dim,output_dim,bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            for i in range(min(input_dim,output_dim)):self.projection.weight[i,i]=1.
        self.alpha=nn.Parameter(torch.tensor(.8));self.log_beta=nn.Parameter(torch.tensor(0.))
    def forward(self,x):return nn.functional.normalize(self.projection(x),dim=-1,eps=1e-8)
    def scores(self,z,a,b):
        d2=((z[a]-z[b])**2).sum(-1)
        return 3*torch.tanh(self.alpha-torch.exp(self.log_beta.clamp(-5,5))*d2)

def correlations(rows,values):
    out={}
    for gender in ('male','female'):
        pos=[i for i,r in enumerate(rows) if r['gender']==gender]
        y=np.array([rows[i]['score'] for i in pos]);p=np.array(values)[pos]
        ok=len(y)>=3 and np.std(y)>0 and np.std(p)>1e-10
        out[gender]=dict(n=len(y),spearman=float(spearmanr(y,p).statistic) if ok else None,
            pearson=float(pearsonr(y,p).statistic) if ok else None,mae=float(np.mean(abs(y-p))) if len(y) else None)
    good=[m['spearman'] for m in out.values() if m['spearman'] is not None]
    out['mean_spearman']=float(np.mean(good)) if len(good)==2 else -1.
    return out

def fit(data,representation,dim,epochs,device,out,resume=False):
    profiles,clips,conf=make_input(data,representation)
    X=pca_transform(profiles,conf).astype('float32')
    tr=[r for r in data['pairs'] if r['split']=='train'];va=[r for r in data['pairs'] if r['split']=='dev']
    x=torch.tensor(X,device=device)
    a=torch.tensor([r['a'] for r in tr],device=device);b=torch.tensor([r['b'] for r in tr],device=device)
    target=torch.tensor([r['score'] for r in tr],device=device,dtype=torch.float32)
    weights=torch.tensor([1/(1+r['variance']/r['n']) for r in tr],device=device,dtype=torch.float32)
    aa=torch.tensor([r['a'] for r in va],device=device);bb=torch.tensor([r['b'] for r in va],device=device)
    # Same observed normal voice, four independent texts; never apply this to falsetto.
    tx=[];ti=[]
    for i,s in enumerate(data['speakers']):
        if data['split'][i]!='train':continue
        for f in data['clips'][s]:tx.append(pca_transform(clips[f][None],conf)[0]);ti.append(i)
    tx=torch.tensor(np.array(tx),device=device,dtype=torch.float32);ti=torch.tensor(ti,device=device)
    model=MetricHead(X.shape[1],dim).to(device);optim=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    start=0;best=-float('inf');history=[];best_state=None
    if resume:
        ck=torch.load(out/'last.pt',map_location=device,weights_only=True)
        if ck['data_hash']!=data['provenance'] or ck['representation']!=representation or ck['output_dim']!=dim:raise ValueError('Resume provenance mismatch')
        model.load_state_dict(ck['model']);optim.load_state_dict(ck['optimizer']);start=ck['epoch']+1
        best=ck['best_dev'];history=ck['history'];best_state=ck['best_model']
    elif out.exists() and any(out.iterdir()):raise ValueError('Run already exists; use --resume')
    out.mkdir(parents=True,exist_ok=True)
    for epoch in range(start,start+epochs):
        model.train();optim.zero_grad();z=model(x);pred=model.scores(z,a,b)
        supervised=(nn.functional.huber_loss(pred,target,reduction='none',delta=1.)*weights).mean()
        repeat=((model(tx)-z[ti])**2).sum(-1).mean()
        loss=supervised+.01*repeat
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optim.step()
        model.eval()
        with torch.no_grad():vp=model.scores(model(x),aa,bb).cpu().numpy()
        val=correlations(va,vp)['mean_spearman']
        history.append(dict(epoch=epoch,loss=float(loss.detach()),dev_spearman=val))
        if val>best:
            best=val;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch
        elif epoch==start:best_epoch=ck.get('best_epoch',start) if resume else epoch
        if epoch%50==0:print(representation,dim,epoch,round(float(loss.detach()),4),'dev',round(val,4),flush=True)
    checkpoint=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optim.state_dict(),
        best_model=best_state,epoch=epoch,best_epoch=best_epoch,best_dev=best,history=history,
        representation=representation,input_dim=X.shape[1],output_dim=dim,preprocessing=conf,data_hash=data['provenance'],
        research_only=True,target='JVS different-person perceived normal-voice similarity; not identity classification',
        edge_policy=data['edge_policy'],seed=20261117,torch_version=str(torch.__version__))
    torch.save(checkpoint,out/'last.pt');checkpoint['model']=best_state;torch.save(checkpoint,out/'best.pt')
    write_json(out/'history.json',history)
    return dict(name=f'{representation}_d{dim}',representation=representation,dim=dim,dev_spearman=best,
        checkpoint=(out/'best.pt').relative_to(ROOT).as_posix(),epoch=best_epoch)

def main():
    p=argparse.ArgumentParser();p.add_argument('--representations',nargs='+',default=['acoustic','ecapa','wavlm3','fusion'])
    p.add_argument('--dimensions',nargs='+',type=int,default=[3,8,16]);p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.epochs<1:p.error('Positive epochs required')
    torch.manual_seed(20261117);np.random.seed(20261117);random.seed(20261117);torch.set_num_threads(2)
    data=load_data();results=[]
    for representation in a.representations:
        for d in a.dimensions:results.append(fit(data,representation,d,a.epochs,a.device,ROOT/f'runs/{representation}_d{d}',a.resume))
    previous=ROOT/'runs/development.json'
    if previous.exists():
        existing=json.loads(previous.read_text());names={r['name'] for r in results};results.extend(r for r in existing['candidates'] if r['name'] not in names)
    chosen=max(results,key=lambda r:r['dev_spearman'])
    write_json(previous,dict(candidates=results,chosen=chosen,data_hash=data['provenance'],test_evaluated=False))
    print('DEVELOPMENT CHOICE',chosen,flush=True)

if __name__=='__main__':main()
