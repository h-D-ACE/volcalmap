"""Compare profile and different-text supervision on the development speakers."""
from runtime import ROOT, write_json
from dataset import load_data, make_input, pca_transform
from train_map import MetricHead, correlations
import numpy as np
import torch


def main():
    data = load_data()
    profiles, clips, conf = make_input(data, 'acoustic')
    def tensor(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device='cuda')
    x = tensor([[pca_transform(clips[f], conf) for f in data['clips'][s]] for s in data['speakers']])
    profile = tensor(pca_transform(profiles, conf))
    tr = [r for r in data['pairs'] if r['split']=='train']
    dev = [r for r in data['pairs'] if r['split']=='dev']
    a,b = ([r[k] for r in tr] for k in ('a','b'))
    da,db = ([r[k] for r in dev] for k in ('a','b'))
    y = tensor([r['score'] for r in tr])
    w = tensor([1/(1+r['variance']/r['n']) for r in tr])
    train = np.flatnonzero(data['split']=='train').tolist()
    pairs = [(i,j) for i in range(4) for j in range(4) if i!=j]
    results = {}
    for condition in ('profile','cross_text'):
        model = MetricHead(32,16).cuda()
        opt = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
        for epoch in range(271):
            opt.zero_grad()
            z = model(x.reshape(-1,32)).reshape(-1,4,16)
            p = model(profile)
            if condition=='profile':
                dist=(p[a]-p[b]).square().sum(-1)[:,None]
            else:
                dist=torch.stack([(z[a,i]-z[b,j]).square().sum(-1) for i,j in pairs],dim=1)
            pred=3*torch.tanh(model.alpha-model.log_beta.clamp(-5,5).exp()*dist)
            loss=(torch.nn.functional.huber_loss(pred,y[:,None].expand_as(pred),reduction='none')*w[:,None]).mean()+.01*(z[train]-p[train,None]).square().sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step()
        with torch.no_grad():
            z=model(x.reshape(-1,32)).reshape(-1,4,16)
            scores=[correlations(dev,-(z[da,i]-z[db,j]).norm(dim=-1).cpu().numpy())['mean_spearman'] for i,j in pairs]
            results[condition]=dict(mean=float(np.mean(scores)),per_assignment=scores)
    write_json(ROOT/'results/cross_text_development.json',results)
    print(results)


if __name__=='__main__':
    main()
