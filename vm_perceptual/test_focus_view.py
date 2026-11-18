import unittest
import numpy as np
from focus_view import focus


class FocusDistanceTests(unittest.TestCase):
    def test_single_axis_and_degenerate_or_duplicate_points(self):
        z=[[0.,0.,0.,0.],[2.,0.,1.,0.],[0.,.5,0.,1.],[0.,0.,0.,0.]]
        records=[dict(record_id=str(i),sample_id=str(i),vector=x,display=x[:2]) for i,x in enumerate(z)]
        index=dict(model_version='a',dimensions=4,records=records,projection_version='view')
        q=dict(model_version='a',sample_id='0',timbre=dict(vector=z[0]))
        for dim in (1,2,3):
            for k in (1,3):
                out=focus(index,q,k,dim)
                for row in out['neighbors']:
                    self.assertEqual(len(row['coordinates']),dim)
                    self.assertAlmostEqual(np.linalg.norm(row['coordinates']),row['distance'])
                self.assertNotIn('0',[r['record_id'] for r in out['neighbors']])


if __name__=='__main__':unittest.main()
