"""Single-WavLM replacement for a three-frontend perceptual teacher; development selection only."""
import argparse
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from transformers import WavLMModel
from runtime import ROOT,file_hash,write_json
from wave_student import batch,load_records
from wave_folds import paired_labels
from perceptual_rank import metrics


class RankUpdate(nn.Module):
    def __init__(self,weight,rank=4):
        super().__init__()
        self.left=nn.Parameter(torch.zeros(weight.shape[0],rank))
        self.right=nn.Parameter(torch.empty(rank,weight.shape[1]))
        nn.init.kaiming_uniform_(self.right,a=math.sqrt(5))

    def forward(self,weight):
        return weight+self.left@self.right


class SSLMap(nn.Module):
    def __init__(self,recipe):
        super().__init__()
        self.backbone=WavLMModel.from_pretrained(str(ROOT.parent/'claude_baselines/models/wavlm'),local_files_only=True)
        self.backbone.requires_grad_(False)
        if recipe=='rank4':
            for layer in self.backbone.encoder.layers[-2:]:
                for block in (layer.attention.q_proj,layer.attention.v_proj):
                    parametrize.register_parametrization(block,'weight',RankUpdate(block.weight))
        self.mix=nn.Parameter(torch.zeros(5))
        self.frame=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128),nn.SiLU())
        self.attention=nn.Linear(128,1)
        self.out=nn.Linear(256,16)

    def forward(self,x,n):
        # Frozen backbone executes without dropout or random layer dropping, also during adaptation.
        self.backbone.eval()
        mask=torch.arange(x.shape[-1],device=x.device)[None,:]<n[:,None]
        hs=self.backbone(x,attention_mask=mask.long(),output_hidden_states=True).hidden_states
        weights=self.mix.softmax(0)
        y=self.frame(sum(w*hs[i] for w,i in zip(weights,(1,3,6,9,12))))
        lengths=self.backbone._get_feat_extract_output_lengths(n)
        valid=torch.arange(y.shape[1],device=x.device)[None,:]<lengths[:,None]
        att=self.attention(y).squeeze(-1).float().masked_fill(~valid,-1e4).softmax(1)[:,:,None]
        y=y.float();mean=(att*y).sum(1)
        sd=((att*y.square()).sum(1)-mean.square()).clamp_min(1e-6).sqrt()
        return F.normalize(self.out(torch.cat((mean,sd),dim=-1)).float(),dim=-1)


def targets_for_records(records,data):
    mapping={}
    result=[]
    for rec in records:
        s=data['speakers'][rec['speaker']]
        if s not in mapping:
            files=sorted((ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit').glob('*.wav'))
            z=np.load(ROOT/'features/rank_map_own'/f'{s}.npy')
            mapping[s]=dict(zip([f.name for f in files],z,strict=True))
        result.append(mapping[s][rec['name']])
    return np.asarray(result)


@torch.inference_mode()
def assess(model,data,records,by,targets,split):
    model.eval()
    names=[s for s in data['speakers'] if data['manifest']['split'][s]==split]
    ids=[idx for s in names for idx in by[data['speakers'].index(s)]]
    labels=paired_labels(data,names,np.arange(len(names)))
    report={}
    for sec in (0,1,2,4):
        rows=[]
        for k in range(0,len(ids),4):
            x,n=batch(records,ids[k:k+4],sec)
            with torch.autocast('cuda',dtype=torch.bfloat16):rows.append(model(x,n).cpu().numpy())
        z=np.concatenate(rows)
        report[str(sec)]={**metrics(z.reshape(len(names),4,-1),labels),'teacher_l2':float(np.linalg.norm(z-targets[ids],axis=1).mean())}
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--recipe',choices=('frozen','rank4'),required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--confirm-test',action='store_true')
    a=ap.parse_args();torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,records,by=load_records();target=targets_for_records(records,data);model=SSLMap(a.recipe).cuda()
    folder=ROOT/'runs'/f'ssl_rank_{a.recipe}_{a.seed}_{a.steps}';folder.mkdir(exist_ok=True)
    teacher=file_hash(ROOT/'runs/rank_map_own.npz')
    learnable={k:p for k,p in model.named_parameters() if p.requires_grad}
    if a.confirm_test:
        ck=torch.load(folder/'best.pt',weights_only=True)
        if ck['teacher']!=teacher:raise ValueError('Teacher changed')
        with torch.no_grad():
            for name,p in learnable.items():p.copy_(ck['state'][name])
        write_json(folder/'test.json',assess(model,data,records,by,target,'test'));return
    if (folder/'best.pt').exists():raise FileExistsError(folder)
    device_target=torch.from_numpy(target).cuda()
    tr=[i for i,s in enumerate(data['speakers']) if data['manifest']['split'][s]=='train']
    opt=torch.optim.AdamW(learnable.values(),lr=.0005,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=.00003)
    best=-np.inf;history=[];tic=time.monotonic()
    for step in range(1,a.steps+1):
        model.train();persons=rng.choice(tr,16,replace=False);ids=[rng.choice(by[p]) for p in persons]
        seconds=float(rng.choice([1,2,3,4,5]))
        x,n=batch(records,ids,seconds,rng)
        valid=torch.arange(x.shape[1],device='cuda')[None,:]<n[:,None]
        amp=x.square().sum(-1,keepdim=True).div(n[:,None]).sqrt()
        db=torch.rand((len(ids),1),device='cuda')*23+12
        x=(x+torch.randn_like(x)*amp*10**(-db/20)*(torch.rand((len(ids),1),device='cuda')<.4))*valid
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x,n)
        y=device_target[ids]
        loss=(z-y).square().sum(-1).mean()+F.mse_loss(z@z.T,y@y.T)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(learnable.values(),3.);opt.step();sched.step()
        if step%150==0 or step==a.steps:
            val=assess(model,data,records,by,target,'dev')
            utility=float(np.mean([v['single_srcc']-.5*v['teacher_l2'] for v in val.values()]))
            history.append(dict(step=step,dev=val,utility=utility,elapsed_seconds=time.monotonic()-tic))
            if utility>best:
                best=utility
                torch.save(dict(state={k:p.detach().cpu().clone() for k,p in learnable.items()},teacher=teacher,
                    seed=a.seed,recipe=a.recipe,step=step,trainable=sum(p.numel() for p in learnable.values()),
                    total=sum(p.numel() for p in model.parameters()),base_revision='4c66d4806a428f2e922ccfa1a962776e232d487b'),folder/'best.pt')
            write_json(folder/'dev.json',history);print(a.recipe,a.seed,step,round(utility,4),round(time.monotonic()-tic),flush=True)


if __name__=='__main__':main()
