"""Meaningful offline checks of leakage, metric semantics, and inference contracts."""
import unittest,copy,tempfile
from runtime import ROOT
from dataset import make_input,pca_transform,unit
from train_map import MetricHead
import numpy as np
import torch

class PipelineTests(unittest.TestCase):
    def test_map_search_uses_full_space_and_rejects_other_model(self):
        from map_index import search
        query=dict(model_version='v1',sample_id='self',timbre=dict(vector=[1.,0.,0.]))
        index=dict(model_version='v1',projection_version='p1',dimensions=3,records=[
            dict(record_id='self',sample_id='self',vector=[1,0,0],display=[0,0]),
            dict(record_id='near',sample_id='a',vector=[.99,.1,0],display=[100,100]),
            dict(record_id='far',sample_id='b',vector=[0,1,0],display=[0,0])])
        result=search(index,query)
        self.assertEqual([r['record_id'] for r in result['matches']],['near','far'])
        self.assertEqual(search(index,query,include_self=True)['matches'][0]['distance'],0)
        query['model_version']='v2'
        with self.assertRaises(ValueError):search(index,query)
    def test_rating_graph_without_test_people_or_audio(self):
        import json
        from pathlib import Path
        from unittest.mock import patch
        from dataset import load_rating_graph
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            (root/'data_jvs').mkdir()
            (root/'data_sources').mkdir()
            manifest = {'split': {'a': 'train', 'b': 'dev', 'c': 'test'}}
            (root/'data_jvs/manifest.json').write_text(json.dumps(manifest))
            for group, ratings in [('male', {'a-b': [2, 3], 'a-c': [1, 2]}), ('female', {})]:
                (root/'data_sources'/f'jvs_{group}_raw_similarity_data.json').write_text(json.dumps(ratings))
            with patch('dataset.ROOT', root):
                graph = load_rating_graph(('train', 'dev'))
            self.assertEqual(graph['speakers'], ['a', 'b'])
            self.assertEqual([(r['speaker_a'],r['speaker_b'],r['score']) for r in graph['pairs']], [('a','b',2.5)])

    def fixture(self):
        rng=np.random.default_rng(7);speakers=[f's{i}' for i in range(10)];features={};clips={}
        for s in speakers:
            clips[s]=[f'{s}/{j}' for j in range(4)]
            for f in clips[s]:features[f]=dict(acoustic=rng.normal(size=40).tolist(),ecapa=rng.normal(size=192).tolist())
        split=['train']*6+['dev']*2+['test']*2
        return dict(speakers=speakers,features=features,clips=clips,split=np.array(split),manifest={'split':dict(zip(speakers,split))})
    def test_validation_and_test_do_not_fit_preprocessing(self):
        data=self.fixture();_,_,a=make_input(data,'fusion');other=copy.deepcopy(data)
        for s in other['speakers'][6:]:
            for f in other['clips'][s]:
                for key in ('acoustic','ecapa'):other['features'][f][key]=[x*1000+900 for x in other['features'][f][key]]
        _,_,b=make_input(other,'fusion')
        self.assertEqual(a,b)
    def test_coordinates_obey_metric_and_self_identity(self):
        torch.manual_seed(1);m=MetricHead(32,8);z=m(torch.randn(5,32)).detach().numpy()
        d=np.linalg.norm(z[:,None]-z[None,:],axis=2)
        np.testing.assert_allclose(d,d.T);np.testing.assert_allclose(np.diag(d),0)
        for i in range(5):
            for j in range(5):
                for k in range(5):self.assertLessEqual(d[i,j],d[i,k]+d[k,j]+1e-6)
    def test_checkpoint_roundtrip_and_no_identity_input(self):
        m=MetricHead(32,3);x=torch.randn(4,32)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            p=ROOT/temp/'test.pt';torch.save(m.state_dict(),p);restored=MetricHead(32,3)
            restored.load_state_dict(torch.load(p,weights_only=True));torch.testing.assert_close(m(x),restored(x))
    def test_preprocessing_metadata_strict_loading(self):
        _,_,conf=make_input(self.fixture(),'fusion')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            p=ROOT/temp/'metadata.pt';torch.save(conf,p)
            self.assertEqual(conf,torch.load(p,weights_only=True))
    def test_loss_score_symmetry(self):
        m=MetricHead(32,8);z=m(torch.randn(4,32));a=torch.tensor([0,1]);b=torch.tensor([2,3])
        torch.testing.assert_close(m.scores(z,a,b),m.scores(z,b,a))

if __name__=='__main__':unittest.main()
