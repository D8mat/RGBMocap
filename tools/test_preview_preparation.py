import sys,tempfile,json,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap import preview_preparation as p


class PreviewPreparation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/p.RESULTS['initial'][0]).write_bytes(b'test')
    def tearDown(self):self.temp.cleanup()
    def test_current_explicit_click_can_reload_without_inference(self):
        with patch.object(p,'result_is_current',return_value=True),patch.object(p,'current',return_value=True),patch.object(p,'surface_current',return_value=True):
            self.assertEqual(p.plan([self.root,self.root],'initial','mamma'),([],1,0))
        w=SimpleNamespace(preview_signature='old',request_people_scene=Mock(),refresh_results=Mock(),
            camera_tools=SimpleNamespace(key='old',sync=Mock()),
            mamma_video_pair=SimpleNamespace(overlay=SimpleNamespace(clear=Mock()),repaint_videos=Mock()))
        p.reload_preview(w)
        self.assertIsNone(w.preview_signature);self.assertIsNone(w.camera_tools.key)
        w.request_people_scene.assert_called_once_with(force=True);w.refresh_results.assert_called_once()
        w.camera_tools.sync.assert_called_once();w.mamma_video_pair.overlay.clear.assert_called_once()
    def test_missing_mesh_rebuilds_even_when_rig_is_current(self):
        with patch.object(p,'result_is_current',return_value=True),patch.object(p,'current',return_value=True):
            jobs,ready,bad=p.plan([self.root],'initial','mamma')
        self.assertEqual([j[0] for j in jobs],['rig']);self.assertEqual((ready,bad),(0,0))
    def test_stale_correction_is_not_replaced_by_original(self):
        (self.root/p.RESULTS['corrected_initial'][0]).write_bytes(b'test')
        with patch.object(p,'result_is_current',return_value=False):
            self.assertEqual(p.plan([self.root],'corrected_initial','mamma'),([],0,1))
    def test_flow_placement_recovery_preserved(self):
        with patch.object(p,'needs_update',return_value=True):
            jobs,_,_=p.plan([self.root],'initial','flowhmr')
        self.assertEqual(len(jobs),1)
    def test_surface_requires_payload_and_receipt(self):
        (self.root/'initial_joints.npz').write_bytes(b'test')
        (self.root/'initial_mesh.json').write_text(json.dumps(dict(version=1,frames_file='mesh.npy')))
        self.assertFalse(p.surface_current(self.root,'initial'))
        (self.root/'mesh.npy').write_bytes(b'test')
        with patch.object(p,'artifacts_current',return_value=False):self.assertFalse(p.surface_current(self.root,'initial'))
        with patch.object(p,'artifacts_current',return_value=True):self.assertTrue(p.surface_current(self.root,'initial'))

if __name__=='__main__':unittest.main()
