"""Face-only sender: real local UDP, stale data and shutdown regression tests."""
from pathlib import Path
import json
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap.face_extract import ARKIT,source_identity
from mocap.face_feed import FaceFeed


class FaceFeedTests(unittest.TestCase):
    def test_udp_identity_and_stop(self):
        with tempfile.TemporaryDirectory() as folder, socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as receiver:
            path=Path(folder);(path/'input.mp4').write_bytes(b'video');(path/'tracking.npz').write_bytes(b'tracking')
            values=np.zeros((2,52),np.float16);values[1,17]=.75
            np.savez(path/'face.npz',values=values,names=np.array(ARKIT),source=np.array(json.dumps(source_identity(path))))
            receiver.bind(('127.0.0.1',0));receiver.settimeout(2)
            with patch('mocap.face_feed.PORT',receiver.getsockname()[1]):
                feed=FaceFeed()
                try:
                    feed.submit(('first',path,1,'Actor',True))
                    packet=json.loads(receiver.recv(8192));self.assertTrue(packet['valid']);self.assertEqual(packet['frame'],1)
                    self.assertEqual(packet['values'][17],.75)
                    (path/'tracking.npz').write_bytes(b'changed tracking')
                    while True:
                        packet=json.loads(receiver.recv(8192))
                        if not packet['valid']:break
                    self.assertEqual(max(packet['values']),0)
                    feed.submit(('stopped',path,1,'Actor',False))
                    packet=json.loads(receiver.recv(8192));self.assertFalse(packet['valid'])
                finally:feed.close()
                self.assertFalse(feed.thread.is_alive())


if __name__=='__main__':unittest.main()
