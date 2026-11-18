import unittest,tempfile
from runtime import ROOT
import numpy as np

class EdgeTests(unittest.TestCase):
    def test_new_edge_policy_and_legacy_inference(self):
        import soundfile as sf
        from extract_features import Frontend
        rate=16000
        t=np.arange(rate)/rate
        x=(.3*np.sin(2*np.pi*160*t)+.1*np.sin(2*np.pi*640*t)).astype('float32')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            dry=ROOT/temp/'dry.wav'
            padded=ROOT/temp/'padded.wav'
            sf.write(dry,x,rate,subtype='FLOAT')
            sf.write(padded,np.pad(x,(rate//2,rate)),rate,subtype='FLOAT')
            front=Frontend(use_ecapa=False,use_wavlm=False,edge_policy='trim_zero')
            np.testing.assert_array_equal(front.extract(dry)['acoustic'],front.extract(padded)['acoustic'])

if __name__=='__main__':unittest.main()
