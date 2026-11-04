"""Synthetic metric camera recovery and invalid-input guards; no AI/GPU."""
import sys, unittest, hashlib, copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from mocap.calibration_solver import solve, lens_points
from mocap.apriltag_detection import dictionary


class CalibrationTests(unittest.TestCase):
    def fixture(self):
        k=np.array([[1100.,0,640],[0,1080,360],[0,0,1]])
        layout=dict(units='metres',world_up='Z',family='tag36h11',markers={})
        for i,(x,y,z) in enumerate([(-.5,-.3,0),(.2,-.3,.1),(-.2,.2,.2),( .4,.2,0)]):
            layout['markers'][str(i)]=np.array([[x,y,z],[x+.15,y,z],[x+.15,y+.15,z],[x,y+.15,z]]).tolist()
        observations=[]; lenses={}; truth={}
        for i in range(2):
            name='camera'+str(i); r=np.array([.15,-.2+i*.25,.04]); t=np.array([i*.4,0,2.5])
            d=np.array([-.05,.01,0,0,0]); markers=[]
            for ident,p in layout['markers'].items():
                pixels=cv2.projectPoints(np.array(p),r,t,k,d)[0].reshape(-1,2)
                markers.append(dict(id=int(ident),corners_px=pixels.tolist()))
            observations.append(dict(camera_id=name,markers=markers,image_size=[1280,720],tag_size_m=.15,family='tag36h11',dictionary_sha256=hashlib.sha256(dictionary().bytesList.tobytes()).hexdigest()))
            lenses[name]=dict(intrinsic_matrix=k.tolist(),distortion_coeffs=d.tolist(),image_size=[1280,720])
            truth[name]=np.column_stack((cv2.Rodrigues(r)[0],t))
        return observations,lenses,layout,truth
    def test_metric_recovery(self):
        obs,lens,layout,truth=self.fixture(); result=solve(obs,lens,layout)
        for key,c in result['cameras'].items():np.testing.assert_allclose(c['extrinsics_matrix'],truth[key],atol=1e-5)
        self.assertTrue(result['requires_undistortion'])
        from mocap.mamma_calibration_export import validate
        validate(result)
    def test_rejections(self):
        obs,lens,layout,_=self.fixture()
        with self.assertRaises(InterruptedError):solve(obs,lens,layout,lambda:True)
        bad=copy.deepcopy(obs); bad[0]['markers'].append(bad[0]['markers'][0])
        with self.assertRaises(ValueError):solve(bad,lens,layout)
        bad=copy.deepcopy(lens);bad['camera0']['image_size']=[640,360]
        with self.assertRaises(ValueError):solve(obs,bad,layout)
        bad=copy.deepcopy(obs);bad[0]['markers']=bad[0]['markers'][:1]
        with self.assertRaises(ValueError):solve(bad,lens,layout)
        bad=copy.deepcopy(obs);bad[0]['markers'][0]['corners_px'][0][0]=float('nan')
        with self.assertRaises(ValueError):solve(bad,lens,layout)
    def test_lens_recovery(self):
        obj=np.zeros((54,3),np.float32);obj[:,:2]=np.mgrid[:9,:6].T.reshape(-1,2)*.025
        k=np.array([[1100.,0,640],[0,1080,360],[0,0,1]]); points=[]
        rng=np.random.default_rng(42)
        for _ in range(20):
            r=rng.uniform(-.45,.45,3);t=np.array([rng.uniform(-.2,.05),rng.uniform(-.15,.05),rng.uniform(.6,1.)])
            points.append(cv2.projectPoints(obj,r,t,k,np.zeros(5))[0].reshape(-1,2))
        result=lens_points(points,[1280,720]); np.testing.assert_allclose(result['intrinsic_matrix'],k,atol=.1)
        with self.assertRaises(ValueError):lens_points(points[:3],[1280,720])
        with self.assertRaises(ValueError):lens_points([points[0]]*12,[1280,720])


if __name__=='__main__':
    cv2.ocl.setUseOpenCL(False);cv2.setNumThreads(2);unittest.main()
