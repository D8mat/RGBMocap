import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from mocap.checkerboard_pose import solve_board,calibration


class BoardTests(unittest.TestCase):
    def fixture(self):
        k=np.array([[1100.,0,640],[0,1090,360],[0,0,1]])
        obj=np.zeros((54,3));obj[:,:2]=np.mgrid[:9,:6].T.reshape(-1,2)*[.025,-.025]
        rot=cv2.Rodrigues(np.array([.3,.1,0]))[0]@np.diag([1.,-1.,-1.])
        t=np.array([-.1,.05,.8]);r=cv2.Rodrigues(rot)[0]
        lens=dict(intrinsic_matrix=k.tolist(),distortion_coeffs=[-.03,.01,0,0,0],image_size=[1280,720])
        p=cv2.projectPoints(obj,r,t,k,np.array(lens['distortion_coeffs']))[0].reshape(-1,2)
        return p,lens,np.column_stack((rot,t))
    def test_recovery_and_parser(self):
        p,lens,truth=self.fixture();c=solve_board(p,lens,[1280,720],9,6,25)
        np.testing.assert_allclose(c['extrinsics_matrix'],truth,atol=1e-5)
        self.assertGreater(c['quality']['camera_center_m'][2],0)
        from mocap.mamma_calibration_export import validate
        validate(calibration(dict(a=c,b=c)))
    def test_bad_inputs(self):
        p,lens,_=self.fixture()
        with self.assertRaises(ValueError):solve_board(p,lens,[640,360],9,6,25)
        with self.assertRaises(ValueError):solve_board(p,lens,[1280,720],6,6,25)
        with self.assertRaises(ValueError):calibration({})
        p[0,0]=float('nan')
        with self.assertRaises(ValueError):solve_board(p,lens,[1280,720],9,6,25)
    def test_reverse_restores_physical_origin(self):
        p,lens,truth=self.fixture()
        detected=p[::-1].copy()
        c=solve_board(detected[::-1].copy(),lens,[1280,720],9,6,25)
        np.testing.assert_allclose(c['extrinsics_matrix'],truth,atol=1e-5)


if __name__=='__main__':unittest.main()
