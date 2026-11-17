"""Offline regression checks; no corpus download or perceptual claim."""
import unittest
from acoustics import np
from formant_tracker import track
from voice_space import VoicePoint,search,validate_contract

class Contracts(unittest.TestCase):
    def test_search(self):
        validate_contract()

    def test_versions_and_missing_style(self):
        q=VoicePoint('q','v1',(0.,))
        with self.assertRaises(ValueError):search(q,[],match_prosody=True)
        with self.assertRaises(ValueError):search(q,[VoicePoint('x','v2',(0.,))])
        with self.assertRaises(ValueError):VoicePoint('x','v1',(float('nan'),))

    def test_tracker_keeps_missing_frames_missing(self):
        f=np.tile([500.,1500.,2500.],(7,5,1))
        b=np.full_like(f,100.)
        f[3]=np.nan
        result,states=track(f,b)
        self.assertTrue(np.isnan(result[3]).all())
        self.assertTrue(np.isfinite(result[[0,1,2,4,5,6]]).all())

    def test_tracker_rejects_transient_broad_pole(self):
        f=np.tile([500.,1500.,2500.],(9,5,1))
        b=np.full_like(f,100.)
        f[4,2,1]=2100.;b[4,2,1]=1000.
        result,states=track(f,b)
        np.testing.assert_allclose(result[:,1],1500.)

if __name__=='__main__':unittest.main()
