"""Camera export rescaling and validation; no models or video decoding."""
import unittest,sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mocap.mamma_calibration_export import original_pixels,validate

class CameraCalibration(unittest.TestCase):
    def setUp(self):
        self.cameras={'cam':dict(intrinsic_matrix=[[500,0,320],[0,510,240],[0,0,1]],extrinsics_matrix=[[1,0,0,-1],[0,1,0,-2],[0,0,1,-3]],image_size=[640,480],distortion_coeffs=[0]*5)}
        self.entries=[dict(id='cam',metadata=dict(width=1920,height=1080))]
    def test_resolution_roundtrip_preserves_extrinsics(self):
        source=copy.deepcopy(self.cameras);data=original_pixels(source,self.entries)
        self.assertEqual(source,self.cameras)
        self.assertEqual(data['cameras']['cam']['intrinsic_matrix'],[[1500,0,960],[0,1147.5,540],[0,0,1]])
        back=original_pixels(data['cameras'],[dict(id='cam',metadata=dict(width=640,height=480))])
        self.assertEqual(back['cameras'],source)
    def test_invalid_rotation_rejected(self):
        self.cameras['cam']['extrinsics_matrix'][0][0]=-1
        with self.assertRaises(ValueError):original_pixels(self.cameras,self.entries)
    def test_name_and_resolution_checked(self):
        data=original_pixels(self.cameras,self.entries)
        with self.assertRaises(ValueError):validate(data,[dict(id='other',metadata=dict(width=1920,height=1080))])
        with self.assertRaises(ValueError):validate(data,[dict(id='cam',metadata=dict(width=640,height=480))])
    def test_nonfinite_rejected(self):
        self.cameras['cam']['intrinsic_matrix'][0][0]=float('nan')
        with self.assertRaises(ValueError):original_pixels(self.cameras,self.entries)

if __name__=='__main__':unittest.main()
