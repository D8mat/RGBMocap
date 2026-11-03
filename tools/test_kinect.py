"""CPU-only protocol, persistence and metric alignment regression checks."""
import io,struct,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mocap.kinect_data import decode,packet,Recording,Session,similarity

def frame(t):
    return struct.pack('<idddi',1,t,t,t,0)+struct.pack('<i',3)+b'jpg'+struct.pack('<i',512*424)+bytes(512*424*2)+struct.pack('<i',0)

class KinectChecks(unittest.TestCase):
    def test_capture_roundtrip_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'capture';r=Recording(folder)
            for t in (17.,17.05,17.2):
                raw=frame(t);r.append(raw,decode(raw))
            r.close();s=Session(folder)
            np.testing.assert_allclose(s.times,[0,.05,.2]);self.assertEqual(s.frame(2)['time'],17.2)
            with self.assertRaises(FileExistsError):Recording(folder)
    def test_truncated_and_oversized_input(self):
        with self.assertRaises(EOFError):decode(frame(1)[:-3])
        with self.assertRaises(ValueError):packet(io.BytesIO(struct.pack('<I',2**31)))
    def test_pending_is_not_a_completed_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'capture';r=Recording(folder)
            with self.assertRaises(ValueError):Session(folder)
            r.close()
            with self.assertRaises(ValueError):Session(folder)
    def test_metric_alignment(self):
        a=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[.2,.3,.4]])
        expected=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        b=a@expected.T*1.7+[.4,-.2,2]
        r,s,t,error=similarity(a,b)
        np.testing.assert_allclose(r,expected,atol=1e-12);self.assertAlmostEqual(s,1.7);self.assertLess(error,1e-12)
        with self.assertRaises(ValueError):similarity(np.zeros((6,3)),np.ones((6,3)))

if __name__=='__main__':unittest.main()
