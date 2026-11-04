"""Metric recovery, ambiguity/input guards and detector-to-board correspondence."""
import copy
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from mocap.apriltag_board import BOARD,layout,fit_lens,floor_pose,correspondences,video_lens
from mocap.apriltag_detection import dictionary,detect


def observation(k,d,r,t,size=(1280,720),partial=False):
    markers=[]
    for ident,obj in layout()['markers'].items():
        if partial and int(ident) in (1,4):continue
        pixels=cv2.projectPoints(np.asarray(obj),r,t,k,d)[0].reshape(4,2)
        markers.append(dict(id=int(ident),corners_px=pixels.tolist()))
    return dict(family='tag36h11',image_size=list(size),markers=markers)


class BoardTests(unittest.TestCase):
    def test_published_definition(self):
        path=Path(__file__).resolve().parents[1]/'docs/samples/apriltag_board_3x2.json'
        self.assertEqual(json.loads(path.read_text()),layout())
    def fixture(self,distorted=False):
        k=np.array([[1000.,0,640],[0,980,360],[0,0,1]])
        d=np.array([-.07,.02,.001,-.001,0]) if distorted else np.zeros(5)
        rng=np.random.default_rng(12);obs=[]
        for i in range(30):
            rot=cv2.Rodrigues(rng.uniform(-.5,.5,3))[0]@np.diag([1.,-1,-1])
            r=cv2.Rodrigues(rot)[0];t=np.array([rng.uniform(-.25,.25),rng.uniform(-.12,.12),rng.uniform(1.1,1.7)])
            obs.append(observation(k,d,r,t,partial=i%3==0))
        return k,d,obs
    def test_lens_and_metric_pose(self):
        k,d,obs=self.fixture()
        lens=fit_lens(obs,True)
        np.testing.assert_allclose(lens['intrinsic_matrix'],k,atol=.1)
        np.testing.assert_array_equal(lens['distortion_coeffs'],np.zeros(5))
        r=cv2.Rodrigues(cv2.Rodrigues(np.array([.3,.12,.02]))[0]@np.diag([1.,-1,-1]))[0]
        t=np.array([.1,-.1,1.4]);pose=floor_pose(observation(k,d,r,t),lens)
        np.testing.assert_allclose(pose['extrinsics_matrix'],np.column_stack((cv2.Rodrigues(r)[0],t)),atol=1e-4)
    def test_real_distortion_is_preserved(self):
        k,d,obs=self.fixture(True);lens=fit_lens(obs)
        np.testing.assert_allclose(lens['intrinsic_matrix'],k,atol=.2)
        np.testing.assert_allclose(lens['distortion_coeffs'],d,atol=.002)
    def test_guards(self):
        k,d,obs=self.fixture()
        with self.assertRaises(ValueError):fit_lens(obs[:3])
        with self.assertRaises(ValueError):fit_lens([obs[0]]*15)
        with self.assertRaises(InterruptedError):fit_lens(obs,cancel=lambda:True)
        bad=copy.deepcopy(obs[0]);bad['markers'].append(bad['markers'][0])
        with self.assertRaises(ValueError):correspondences(bad)
        bad=copy.deepcopy(obs[0]);bad['markers'][0]['id']=19
        with self.assertRaises(ValueError):correspondences(bad)
        with self.assertRaises(ValueError):video_lens('unused',0,0,0)
    def test_texture_matches_corner_order(self):
        im=np.full((1560,2280),255,np.uint8)
        for ident in BOARD['ids']:
            row,col=divmod(ident,3);x,y=(30+col*180)*4,(30+row*180)*4
            im[y:y+600,x:x+600]=cv2.aruco.generateImageMarker(dictionary(),ident,600)
        obs,_=detect(im);self.assertEqual(sorted(m['id'] for m in obs['markers']),list(range(6)))
        obj,img,_=correspondences(obs)
        expected=np.column_stack(((obj[:,0]+.285)*4000,(.195-obj[:,1])*4000))
        np.testing.assert_allclose(img,expected,atol=2)


if __name__=='__main__':unittest.main()
