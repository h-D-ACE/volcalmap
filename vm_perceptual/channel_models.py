"""Deterministic synthetic recording channels. Pitch and speech rate are preserved."""
import torch
from torch.nn import functional as F


def apply_channel(x,n,kind,seed):
    if kind=='clean':return x
    gen=torch.Generator(device=x.device).manual_seed(int(seed))
    b,l=x.shape;sr=16000
    def uniform(lo,hi,shape):return lo+(hi-lo)*torch.rand(shape,device=x.device,generator=gen)
    mask=torch.arange(l,device=x.device)[None,:]<n[:,None]
    x=x*mask
    nfft=1<<(l+8000-1).bit_length()
    freq=torch.fft.rfftfreq(nfft,1/sr,device=x.device)[None]
    original_level=(x.square().sum(-1,keepdim=True)/n[:,None]).clamp_min(1e-12).sqrt()
    spec=torch.fft.rfft(x,nfft)
    if kind in ('equalizer','both','telephone'):
        if kind=='telephone':
            hp=300.;lp=3400.;slope=torch.zeros((b,1),device=x.device)
        else:
            hp=uniform(35,160,(b,1));lp=uniform(4400,7600,(b,1));slope=uniform(-3.5,3.5,(b,1))
        f=freq.clamp_min(1)
        response=(1+(hp/f)**6).rsqrt()*(1+(f/lp)**10).rsqrt()*10**((slope*torch.log2(f.clamp_min(80)/1000)).clamp(-14,14)/20)
        spec=spec*response
    if kind in ('room','both','late_room'):
        length=6400 if kind=='late_room' else 3400
        t=torch.arange(length,device=x.device)[None,:]/sr
        rt=uniform(.35,.58,(b,1)) if kind=='late_room' else uniform(.08,.32,(b,1))
        impulse=torch.randn((b,length),device=x.device,generator=gen)*torch.exp(-6.908*t/rt)
        impulse[:,:100]=0;impulse=F.normalize(impulse,dim=1)*uniform(.2,.65,(b,1))
        for _ in range(5):
            delay=torch.randint(80,1250,(b,1),device=x.device,generator=gen)
            impulse.scatter_add_(1,delay,uniform(-.22,.31,(b,1)))
        impulse[:,0]=1
        spec=spec*torch.fft.rfft(impulse,nfft)
    y=torch.fft.irfft(spec,nfft)[:,:l]*mask
    if kind in ('pink_noise','both'):
        noise=torch.randn(x.shape,device=x.device,generator=gen)
        nf=torch.fft.rfft(noise,nfft)*(freq.clamp_min(60)/1000)**-.5
        v=torch.fft.irfft(nf,nfft)[:,:l]*mask
        scale=(v.square().sum(-1,keepdim=True)/n[:,None]).clamp_min(1e-12).rsqrt()
        db=uniform(12,28,(b,1))
        y=y+v*scale*original_level*10**(-db/20)
    if kind=='telephone':
        peak=y.abs().amax(1,keepdim=True).clamp_min(1e-8)
        v=(y/peak).clamp(-1,1)
        comp=v.sign()*torch.log1p(255*v.abs())/5.545177
        comp=((comp+1)*127.5).round()/127.5-1
        y=comp.sign()*torch.expm1(5.545177*comp.abs())/255*peak
    gain=(y.square().sum(-1,keepdim=True)/n[:,None]).clamp_min(1e-12).rsqrt()*original_level
    return y*gain*mask
