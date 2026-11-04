"""CPU-only integrity and cancellation contracts for Unreal export checkpoints."""
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap import unreal_checkpoint as c

class Checkpoints(unittest.TestCase):
    def test_integrity_and_incomplete_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            self.assertIsNone(c.valid(p,'prepared'))
            (p/'motion').write_bytes(b'complete')
            c.complete(p,'prepared',['motion'],{'frames':2})
            self.assertEqual(c.valid(p,'prepared'),{'frames':2})
            (p/'motion').write_bytes(b'partial')
            self.assertIsNone(c.valid(p,'prepared'))
    def test_exclusive_lock_released_on_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaises(ValueError):
                with c.locked(p):
                    with self.assertRaises((RuntimeError,BlockingIOError)):
                        with c.locked(p):pass
                    raise ValueError('cancelled')
            with c.locked(p):pass
    def test_publication_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'cached').write_bytes(b'new');(p/'target').write_bytes(b'old')
            c.publish(p/'cached',p/'target')
            self.assertEqual((p/'target').read_bytes(),b'new')
            self.assertEqual((p/'cached').read_bytes(),b'new')
            self.assertEqual(len(list(p.iterdir())),2)
    def test_identity_tracks_inputs(self):
        self.assertEqual(c.identity({'a':1,'b':2}),c.identity({'b':2,'a':1}))
        self.assertNotEqual(c.identity({'a':1}),c.identity({'a':2}))

if __name__=='__main__':unittest.main()
