import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from common import assert_data, assert_source_contract, config, data_files, digest, new_output, sources

class ContractGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ref = self.root / 'reference'
        self.ref.mkdir()
        (self.ref / 'train.py').write_text('original')
        (self.ref / 'utils').mkdir()
        (self.ref / 'utils/loss_utils.py').write_text('original loss')
        self.cand = self.root / 'candidate'
        shutil.copytree(self.ref, self.cand)
    def tearDown(self):
        self.tmp.cleanup()
    def test_candidate_scope(self):
        (self.cand / 'utils/loss_utils.py').write_text('equivalent implementation')
        self.assertEqual(assert_source_contract(self.cand, self.ref)['changed'], ['utils/loss_utils.py'])
        (self.cand / 'train.py').write_text('forbidden')
        with self.assertRaisesRegex(ValueError, 'outside allowed'):
            assert_source_contract(self.cand, self.ref)
    def test_escaping_source_link(self):
        (self.cand / 'link.py').symlink_to(self.ref / 'train.py')
        with self.assertRaisesRegex(ValueError, 'escapes tree'):
            sources(self.cand)
    def test_gui_and_knn_binary_fixed(self):
        for rel in ['gaussian_renderer/network_gui.py', 'submodules/simple-knn/simple_knn/_C.so']:
            target = self.cand / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'forbidden change')
            with self.assertRaisesRegex(ValueError, 'outside allowed'):
                assert_source_contract(self.cand, self.ref)
            target.unlink()
    def test_unique_outputs(self):
        new_output(self.root / 'run')
        with self.assertRaises(FileExistsError):
            new_output(self.root / 'run')
    def test_data_and_reference_mutations(self):
        scene = self.root / 'bonsai'
        (scene / 'images_2').mkdir(parents=True)
        (scene / 'images_2/img.jpg').write_bytes(b'image')
        (scene / 'sparse/0').mkdir(parents=True)
        (scene / 'sparse/0/points3D.ply').write_bytes(b'prepared public derived PLY')
        c = {'reference': str(self.ref), 'scenes': {'bonsai': {'path': str(scene), 'images': 'images_2'}}, 'data_manifest': str(self.root / 'data.json')}
        records = data_files(c, 'bonsai')
        Path(c['data_manifest']).write_text(json.dumps({'scenes': {'bonsai': records}, 'reference_sources': sources(self.ref)}))
        self.assertEqual(assert_data(c, 'bonsai'), digest(records))
        (scene / 'images_2/img.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'data changed'):
            assert_data(c, 'bonsai')
        (scene / 'images_2/img.jpg').write_bytes(b'image')
        (self.ref / 'train.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'reference source changed'):
            assert_data(c, 'bonsai')
    def test_native_schedule_cannot_drift(self):
        example = Path(__file__).resolve().parents[1] / 'task/config.example.json'
        c = json.loads(example.read_text())
        c['schedule']['iterations'] = 100
        p = self.root / 'bad.json'
        p.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError, 'native schedule'):
            config(p)

if __name__ == '__main__':
    unittest.main()
