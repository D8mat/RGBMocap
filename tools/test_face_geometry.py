"""Geometry regression tests: close-up crop, optional tracking and exact masks."""
import sys,json,time,tempfile
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap.face_extract import detection_crops,source_identity
from mocap.face_data import FaceCache
from mocap.face_regions import RegionReader


class FaceGeometryTests(unittest.TestCase):
    def test_closeup_and_optional_tracking(self):
        rows=detection_crops([3,0,2159,3824],2160,3840)
        self.assertEqual(rows[0],('person',(3,0,2159,3824)))
        self.assertEqual(detection_crops(None,2160,3840),[('frame',(0,0,2160,3840))])
        self.assertEqual(detection_crops([10,0,1,5],20,20),[])
    def test_reader_and_missing_detections(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'input.mp4').write_bytes(b'video');identity=source_identity(p)
            self.assertIsNone(identity['tracking.npz'])
            points=np.zeros((2,478,2),np.float16);points[1]=np.nan
            np.savez(p/'face.npz',values=np.zeros((2,52),np.float16),names=np.array(__import__('mocap.face_extract',fromlist=['ARKIT']).ARKIT),
                     valid=np.array([True,False]),regions=np.array([[.1,.1,.9,.9],[np.nan]*4]),landmarks=points,source=np.array(json.dumps(identity)))
            cache=FaceCache();cache.frame(p,0);self.assertEqual(cache.last_status,'Ready')
            cache.frame(p,1);self.assertIn('No face detected',cache.last_status)
            reader=RegionReader()
            try:
                reader.submit(('token',p));end=time.monotonic()+3
                while reader.output is None and time.monotonic()<end:time.sleep(.02)
                self.assertEqual(reader.output[0],'token');self.assertEqual(reader.output[2].shape,(2,478,2))
                self.assertEqual(reader.output[3].tolist(),[True,False])
                (p/'input.mp4').write_bytes(b'changed video');end=time.monotonic()+3
                while not reader.output[-1] and time.monotonic()<end:time.sleep(.02)
                self.assertIn('stale',reader.output[-1])
            finally:reader.close()
            self.assertFalse(reader.thread.is_alive())


if __name__=='__main__':unittest.main()
