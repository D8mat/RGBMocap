"""Face cache, gap policy and negotiated stream regressions (no model required)."""
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap.face_extract import ARKIT, _hold, source_identity
from mocap.face_data import FaceCache
from mocap.bone_stream import BoneStream, StreamRig


class FaceTests(unittest.TestCase):
    def test_gap_policy(self):
        values = np.zeros((12, 52), np.float32)
        valid = np.zeros(12, bool); valid[[1, 3, 10]] = True
        values[1] = .2; values[3] = .4; values[10] = .8
        _hold(values, valid, max_gap=2)
        self.assertAlmostEqual(float(values[2, 0]), .2)
        self.assertTrue(np.all(values[4:10] == 0))
        self.assertTrue(np.all(values[[0, 11]] == 0))

    def make_take(self, path, value):
        (path/'input.mp4').write_bytes(b'video')
        (path/'tracking.npz').write_bytes(b'tracking')
        np.savez(path/'face.npz', values=np.full((2,52),value,np.float16),
                 names=np.array(ARKIT), source=np.array(json.dumps(source_identity(path))))

    def test_stale_and_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);self.make_take(p,.5);cache=FaceCache()
            self.assertEqual(cache.frame(p,1)[0],.5)
            self.assertEqual(cache.frame(p,2)[0],0)
            (p/'tracking.npz').write_bytes(b'changed tracking')
            self.assertEqual(cache.frame(p,1)[0],0)
            (p/'face.npz').write_bytes(b'invalid')
            self.assertEqual(cache.frame(p,1)[0],0)

    def test_socket_negotiation_and_frame_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);a=p/'a';b=p/'b';a.mkdir();b.mkdir()
            self.make_take(a,.25);self.make_take(b,.75)
            rig=StreamRig(['Hips'],[-1],np.zeros((1,3)),np.eye(3)[None],np.eye(3)[None],np.ones(1))
            sender=BoneStream(port=0)
            client=socket.create_connection(sender.listener.getsockname());client.settimeout(3)
            def read():
                def exact(n):
                    data=b''
                    while len(data)<n:
                        chunk=client.recv(n-len(data))
                        if not chunk:raise RuntimeError('Disconnected')
                        data+=chunk
                    return data
                size,=struct.unpack('<I',exact(4));raw=exact(size)
                return raw[:1],raw[1:]
            try:
                actors=(('a'*64,'A',rig,0,np.eye(4)[None],None,a),
                        ('b'*64,'B',rig,1,np.eye(4)[None],None,b))
                sender.publish(actors,('a'*64,'b'*64))
                schemas={}
                while True:
                    tag,data=read()
                    if tag==b'R':
                        schema=json.loads(data);schemas[schema['slot']]=schema
                    if tag==b'S' and struct.unpack_from('<H',data,8)[0]==2:break
                # Old receiver receives no face or selection extension.
                self.assertEqual(len(data),10+2*(8+48))
                client.sendall(b'VQ')
                while True:
                    tag,data=read()
                    if tag==b'S' and len(data)>10+112+66:break
                offset=10+112
                total,=struct.unpack_from('<H',data,offset);self.assertEqual(total,2)
                offset+=2+total*32
                total,=struct.unpack_from('<H',data,offset);self.assertEqual(total,2);offset+=2
                for expected in (.25,.75):
                    slot,=struct.unpack_from('<I',data,offset)
                    values=struct.unpack_from('<52f',data,offset+4)
                    self.assertEqual(values[0],expected);offset+=212
                self.assertEqual(offset,len(data))
                if '--save-fixture' in sys.argv:
                    dest=Path(sys.argv[sys.argv.index('--save-fixture')+1]);dest.mkdir(exist_ok=True)
                    (dest/'snapshot.bin').write_bytes(data)
                    (dest/'schemas.json').write_text(json.dumps(schemas))
            finally:
                client.close();sender.close()


if __name__=='__main__':
    unittest.main(argv=[sys.argv[0]])
