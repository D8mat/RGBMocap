import io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mocap.character_position import matrix,offset,recenter_offset,apply,FILE
from mocap.scene_export import context,transform_motion


class PositionTests(unittest.TestCase):
    def test_preview_recenter_repeat_reset_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=np.array([[[3.,1.2,-4.]],[[4.,1.4,-3.]]])
            view=SimpleNamespace(joints=original.copy(),index=0,single_transform=np.eye(4),update=lambda:None)
            status=[]
            w=SimpleNamespace(project=root,busy=False,model_choice=SimpleNamespace(currentData=lambda:'flowhmr'),
                motion_view=view,selected_character=lambda:None,scene_data=None,scene_cache={},preview_data=None,
                request_people_scene=lambda **kw:None,status=SimpleNamespace(setText=status.append))
            linked=dict(takes=[dict(source=str(root),transform=np.eye(4).tolist())])
            w.characters=[linked];w.save_characters=lambda:None
            apply(w);np.testing.assert_allclose(original[0,0]+offset(root),[0,1.2,0])
            np.testing.assert_array_equal(view.joints,original)
            np.testing.assert_allclose(np.array(linked['takes'][0]['transform'])[:3,3],offset(root))
            first=offset(root);apply(w);np.testing.assert_allclose(offset(root),first)
            shifted=original+offset(root);np.testing.assert_allclose(shifted[1]-shifted[0],original[1]-original[0])
            self.assertEqual(context(root,root)['transform'],matrix(root).tolist())
            apply(w,True);np.testing.assert_allclose(offset(root),0)
    def test_bvh_export_keeps_height_rotation_and_motion(self):
        header=b'HIERARCHY\nROOT hips\n{\nCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n'
        header+=b'CHANNELS 3 Zrotation Xrotation Yrotation\n'*51+b'}\nMOTION\nFrames: 2\nFrame Time: 0.0333333333\n'
        values=np.zeros((2,159));values[:,:6]=[[3,1.2,-4,20,0,0],[4,1.4,-3,25,0,0]]
        source=header+b''.join((' '.join(map(str,row))+'\n').encode() for row in values)
        out=io.BytesIO();t=np.eye(4);t[:3,3]=[-3,0,4]
        transform_motion(io.BytesIO(source),out,t)
        actual=np.array([list(map(float,l.split())) for l in out.getvalue().splitlines()[-2:]])
        np.testing.assert_allclose(actual[:,:3],[[0,1.2,0],[1,1.4,1]],atol=1e-5)
        np.testing.assert_allclose(actual[:,3:],values[:,3:],atol=1e-5)
    def test_reject_invalid_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/FILE).write_text(json.dumps(dict(offset=[0,1,0])))
            with self.assertRaises(ValueError):matrix(root)
        with self.assertRaises(ValueError):recenter_offset([0,0,0],[float('nan'),1,0])


if __name__=='__main__':unittest.main()
