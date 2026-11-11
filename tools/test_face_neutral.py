"""Neutral-frame calibration persistence, raw preservation and stream regressions."""
import hashlib,json,socket,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap.face_extract import ARKIT,source_identity
from mocap.face_data import FaceCache
from mocap.face_neutral import save_reference
from mocap.face_feed import FaceFeed


class NeutralTests(unittest.TestCase):
    def setup_take(self,p):
        (p/'input.mp4').write_bytes(b'video')
        values=np.tile(np.array([.25,.75,.125,1.],np.float16)[:,None],(1,52))
        np.savez(p/'face.npz',values=values,names=np.array(ARKIT),valid=np.array([True,True,False,True]),source=np.array(json.dumps(source_identity(p))))
    def test_zero_raw_reselection_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.setup_take(p);before=(p/'face.npz').read_bytes();cache=FaceCache()
            self.assertEqual(cache.frame(p,1)[0],.75)
            save_reference(p,0)
            self.assertTrue(np.array_equal(cache.frame(p,0),np.zeros(52)))
            self.assertTrue(np.all(cache.frame(p,1)==.5))
            save_reference(p,1)
            self.assertTrue(np.all(cache.frame(p,1)==0));self.assertTrue(np.all(cache.frame(p,0)==0))
            self.assertTrue(np.all(cache.frame(p,3)==.25))
            save_reference(p,3);self.assertTrue(np.all(cache.frame(p,3)==0))
            save_reference(p,reset=True);self.assertTrue(np.all(cache.frame(p,1)==.75))
            self.assertEqual((p/'face.npz').read_bytes(),before)
    def test_invalid_frame_and_recapture(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.setup_take(p);save_reference(p,0);before=(p/'face_neutral.json').read_bytes()
            for frame in [-1,2,4]:
                with self.assertRaises(ValueError):save_reference(p,frame)
            self.assertEqual((p/'face_neutral.json').read_bytes(),before)
            cache=FaceCache();self.assertTrue(np.all(cache.frame(p,0)==0))
            stat=(p/'face.npz').stat();__import__('os').utime(p/'face.npz',ns=(stat.st_atime_ns,stat.st_mtime_ns+10000000))
            self.assertTrue(np.all(cache.frame(p,0)==.25));self.assertIn('capture changed',cache.calibration_status)
    def test_independent_takes_and_udp_zero(self):
        with tempfile.TemporaryDirectory() as tmp,socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as receiver:
            a=Path(tmp)/'a';b=Path(tmp)/'b';a.mkdir();b.mkdir();self.setup_take(a);self.setup_take(b)
            save_reference(a,0);cache=FaceCache()
            self.assertTrue(np.all(cache.frame(a,0)==0));self.assertTrue(np.all(cache.frame(b,0)==.25))
            receiver.bind(('127.0.0.1',0));receiver.settimeout(2)
            with patch('mocap.face_feed.PORT',receiver.getsockname()[1]):
                feed=FaceFeed()
                try:
                    feed.submit(('t',a,0,'Actor',True));packet=json.loads(receiver.recv(8192))
                    self.assertTrue(packet['valid']);self.assertEqual(packet['values'],[0.]*52)
                finally:feed.close()


if __name__=='__main__':unittest.main()
