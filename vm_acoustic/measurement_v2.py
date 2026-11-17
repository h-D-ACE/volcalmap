"""Independent feature validity, context-aware vowel measurement and uncertainty.

Unlike v1, no whole-utterance 20-joint-frame gate. Missing quantities stay masked.
"""
from math import gcd
from acoustics import np,pm,call,FEATURES
from scipy.signal import resample_poly

class Measurement:
    def __init__(self,y,sr,ceiling=5500,window=.04):
        g=gcd(sr,24000)
        y=resample_poly(y,24000//g,sr//g) if sr!=24000 else np.asarray(y)
        y=y-np.mean(y)
        self.sound=pm.Sound(y,sampling_frequency=24000)
        self.duration=len(y)/24000
        self.pitch=self.sound.to_pitch_ac(time_step=.005,pitch_floor=60,pitch_ceiling=600,
                         voicing_threshold=.45,silence_threshold=.03)
        self.formants=self.sound.to_formant_burg(time_step=.005,max_number_of_formants=5,
                         maximum_formant=ceiling,window_length=window,pre_emphasis_from=50)
        self.hnr=self.sound.to_harmonicity_cc(time_step=.005,minimum_pitch=60,silence_threshold=.1,periods_per_window=4.5)
        self.cpp=call(self.sound,"To PowerCepstrogram",60,.005,5000,50)
        self.cache={}

    def frame(self,t):
        t=float(t)
        key=round(t,7)
        if key in self.cache:return self.cache[key]
        out=np.full(6,np.nan)
        f0=self.pitch.get_value_at_time(t)
        if np.isfinite(f0) and f0>0:out[0]=np.log(f0)
        for j in range(1,4):
            f=self.formants.get_value_at_time(j,t)
            if np.isfinite(f) and 100<f<5000:out[j]=np.log(f)
        # Estimate source-related features only when voicing is available;
        # preserve formants separately if the source estimate is unavailable.
        if np.isfinite(out[0]):
            h=self.hnr.get_value(t)
            if np.isfinite(h) and h>-100:out[5]=h
            try:
                cep=call(self.cpp,"To PowerCepstrum (slice)",t)
                c=call(cep,"Get peak prominence",60,600,"Parabolic",.001,0.,"Straight","Robust")
                if np.isfinite(c):out[4]=c
            except pm.PraatError:pass
        self.cache[key]=out
        return out

    def token(self,start,end,shift=0):
        if not 0<=start<end<=self.duration+.02:
            return dict(values=[None]*6,counts=[0]*6,mask=[False]*6,flags=["invalid_interval"])
        duration=end-start
        # Sample the central 60%, preserving changing trajectories instead of
        # choosing a local region that happens to minimize microphone distance.
        n=max(3,int(duration*.6/.01)+1)
        tt=np.linspace(start+.2*duration,end-.2*duration,n)+shift
        tt=tt[(tt>=.03)&(tt<=self.duration-.03)]
        if not len(tt):return dict(values=[None]*6,counts=[0]*6,mask=[False]*6,flags=["edge_interval"])
        v=np.array([self.frame(t) for t in tt])
        values=[];counts=[];iqr=[]
        for j in range(6):
            a=np.sort(v[np.isfinite(v[:,j]),j]);counts.append(len(a))
            cut=int(len(a)*.1)
            aa=a[cut:len(a)-cut] if cut else a
            values.append(float(np.mean(aa)) if len(a)>=3 else None)
            iqr.append(float(np.percentile(a,75)-np.percentile(a,25)) if len(a)>=3 else None)
        flags=[]
        if duration<.12:flags.append("short_token")
        if duration<.18:flags.append("not_long_reference_token")
        if any(c<3 for c in counts):flags.append("partial_features")
        if all(values[j] is not None for j in (1,2,3)) and not values[1]<values[2]<values[3]:flags.append("formant_order_conflict")
        return dict(values=values,counts=counts,mask=[q is not None for q in values],iqr=iqr,
                    duration=duration,times=tt.tolist(),flags=flags)
