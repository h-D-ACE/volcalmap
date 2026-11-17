"""Multi-ceiling path continuity diagnostic. No vowel-identity templates."""
from acoustics import np,pm

CEILINGS=(4500,5000,5500,6000,6500)

class CandidateBank:
    def __init__(self,y,sr):
        sound=pm.Sound(y,sampling_frequency=sr)
        self.models=[sound.to_formant_burg(time_step=.005,max_number_of_formants=5,
            maximum_formant=c,window_length=.025,pre_emphasis_from=50) for c in CEILINGS]
    def at(self,times):
        ff=[];bb=[]
        for f in self.models:
            ff.append([[f.get_value_at_time(i,float(t)) for i in (1,2,3)] for t in times])
            bb.append([[f.get_bandwidth_at_time(i,float(t)) for i in (1,2,3)] for t in times])
        return np.transpose(np.array(ff),(1,0,2)),np.transpose(np.array(bb),(1,0,2))

def candidates(y,sr,times):return CandidateBank(y,sr).at(times)

def track(freq,bw,strength=1):
    # candidate state = complete F1/F2/F3 from one LPC setting. No swapping
    # individual indices across models to fabricate a nonphysical triple.
    valid=np.isfinite(freq).all(2)&np.isfinite(bw).all(2)&(freq[:,:,0]>100)&(freq[:,:,2]<5500)&(bw>0).all(2)
    safe=np.where(np.isfinite(freq)&(freq>0),freq,100.)
    band=np.where(np.isfinite(bw)&(bw>0),bw,5000.)
    emission=np.sum(np.log1p((band/300)**2)+np.maximum((35-band)/35,0)**2,axis=2)
    emission+=.02*((np.array(CEILINGS)-5500)/1000)**2
    emission[~valid]=1e6
    n,k,_=freq.shape
    cost=emission[0].copy();parents=np.zeros((n,k),int)
    for t in range(1,n):
        change=np.mean(((np.log(safe[t-1,:,None,:])-np.log(safe[t,None,:,:]))/.05)**2,axis=2)
        penalty=strength*np.minimum(change,100)+.02*(np.arange(k)[:,None]!=np.arange(k)[None,:])
        combined=cost[:,None]+penalty
        parents[t]=combined.argmin(0)
        cost=combined.min(0)+emission[t]
    states=[int(cost.argmin())]
    for t in range(n-1,0,-1):states.append(int(parents[t,states[-1]]))
    states=np.array(states[::-1]);result=freq[np.arange(n),states].copy()
    result[~valid[np.arange(n),states]]=np.nan
    return result,states
