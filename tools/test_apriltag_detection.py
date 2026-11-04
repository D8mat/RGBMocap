"""AprilTag dictionary IDs, corners, rotation, duplicates and empty frames."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from mocap.apriltag_detection import detect,dictionary

class AprilTags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cv2.ocl.setUseOpenCL(False);cv2.setNumThreads(2)
    def tag(self,ident=3):
        image=np.full((400,400),255,np.uint8);image[80:320,80:320]=cv2.aruco.generateImageMarker(dictionary(),ident,240);return image
    def test_id_and_rotation(self):
        for n in range(4):
            result,_=detect(np.ascontiguousarray(np.rot90(self.tag(),n)))
            self.assertEqual([m['id'] for m in result['markers']],[3]);self.assertEqual(result['tag_size_m'],.15)
            p=np.array(result['markers'][0]['corners_px']);self.assertTrue(np.allclose(p.mean(0),[199.5,199.5],atol=1))
    def test_duplicate_flagged(self):
        result,_=detect(np.hstack([self.tag(),self.tag()]))
        self.assertEqual(len(result['markers']),2);self.assertTrue(all(m['duplicate_id'] for m in result['markers']))
    def test_empty_and_invalid_size(self):
        result,_=detect(np.full((400,400),255,np.uint8));self.assertEqual(result['markers'],[])
        with self.assertRaises(ValueError):detect(self.tag(),0)
    def test_perspective_corners(self):
        source=self.tag();a=np.float32([[0,0],[399,0],[399,399],[0,399]]);b=np.float32([[30,20],[390,50],[370,390],[10,350]])
        h=cv2.getPerspectiveTransform(a,b);im=cv2.warpPerspective(source,h,(420,420),borderValue=255)
        result,_=detect(im);self.assertEqual([m['id'] for m in result['markers']],[3])
        expected=cv2.perspectiveTransform(np.float32([[[80,80],[319,80],[319,319],[80,319]]]),h)[0]
        actual=np.array(result['markers'][0]['corners_px'])
        self.assertLess(max(min(np.linalg.norm(actual-p,axis=1)) for p in expected),2)

if __name__=='__main__':unittest.main()
