import sys,unittest,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap.retarget.foot_lock import solve
from mocap.retarget.solver import fk,PARENTS
class FootLockTests(unittest.TestCase):
 def base(self):
  rest=np.zeros((22,3));rest[0]=[0,1,0]
  for j in range(1,22):rest[j]=rest[PARENTS[j]]+[0,.1,0]
  for h,k,a,t,x in [(1,4,7,10,-.1),(2,5,8,11,.1)]:
   rest[h]=[x,.9,0];rest[k]=[x,.5,.05];rest[a]=[x,.1,0];rest[t]=[x,.1,.15]
  local=np.tile(np.eye(3),(40,22,1,1));root=np.tile(rest[0],(40,1));root[:,1]+=.03*np.sin(np.arange(40)/8)
  points,world=fk(local,rest,root)
  return dict(rest=rest,root=root,local=local,world=world,points=points)
 def solve_case(self, lift=False):
  b=self.base();n=len(b['root']);geo={}
  for side,j in enumerate((7,8)):
   geo[f'foot_v{side}']=b['rest'][j][None]+[[0,-.08,0]]
   w=np.zeros((1,22));w[0,j]=1;geo[f'foot_w{side}']=w
  height=np.zeros((n,2))
  if lift:height[10:25,0]=.06*np.sin(np.linspace(0,np.pi,15))
  src=dict(points=b['points'].copy(),origin=np.zeros(3),foot_surface_height=height)
  src['points'][:,[7,8],0]=b['points'][0,[7,8],0]
  src['points'][:,[7,8],2]=b['points'][0,[7,8],2]
  out,r=solve(b,src,geo,np.ones((n,4),bool))
  return b,out,r,geo,height
 def test_surface_height_and_lengths(self):
  from mocap.retarget.foot_lock import sole_heights
  b,out,r,geo,height=self.solve_case()
  np.testing.assert_allclose(sole_heights(out,geo),height,atol=1e-5)
  length=np.linalg.norm(out['points'][:,1:]-out['points'][:,PARENTS[1:]],axis=-1)
  expected=np.linalg.norm(b['rest'][1:]-b['rest'][PARENTS[1:]],axis=-1)
  np.testing.assert_allclose(length,np.broadcast_to(expected,length.shape),atol=1e-10)
 def test_lift_even_inside_contact(self):
  from mocap.retarget.foot_lock import sole_heights
  b,out,r,geo,height=self.solve_case(True)
  np.testing.assert_allclose(sole_heights(out,geo),height,atol=1e-5)
  self.assertGreater(sole_heights(out,geo)[:,0].max(),.05)
 def test_foot_rotation_preserved(self):
  b,out,r,geo,height=self.solve_case(True)
  np.testing.assert_allclose(out['world'][:,[7,8,10,11]],b['world'][:,[7,8,10,11]],atol=1e-12)
 def test_body_rotation_preserved(self):
  b,out,r,geo,height=self.solve_case(True)
  np.testing.assert_allclose(out['world'][:,[0,3,6,9,12,15,16,17]],b['world'][:,[0,3,6,9,12,15,16,17]],atol=1e-12)
 def test_source_knee_flexion(self):
  # Constant stance isolates proportion fitting from temporal transition smoothing.
  from unittest.mock import patch
  static=self.base();static['root'][:]=static['root'][0]
  static['points'],static['world']=fk(static['local'],static['rest'],static['root'])
  with patch.object(self,'base',return_value=static):b,out,r,geo,height=self.solve_case()
  def angle(p):
   u=p[:,[4,5]]-p[:,[1,2]];v=p[:,[7,8]]-p[:,[4,5]]
   return np.arccos(np.clip((u*v).sum(-1)/(np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)),-1,1))
  np.testing.assert_allclose(angle(out['points']),angle(b['points']),atol=.03)
  self.assertGreater(r['pelvis_compensation_max_mm'],1.)
if __name__=='__main__':unittest.main()

