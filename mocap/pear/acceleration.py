"""Validated full-mesh live path: FP16 encoder and FP32 decoder/geometry."""
import torch,types,ast,copy
from pathlib import Path
from ..pear_runtime import engine_folder
ROOT=engine_folder().parent
PROFILES={'fast':dict(sdpa=True,precision='fp16',graph=True,geometry='full',fast_fk=True)}

def sdpa_forward(self,x):
    b,n,c=x.shape;qkv=self.qkv(x).reshape(b,n,3,self.num_heads,-1).permute(2,0,3,1,4)
    q,k,v=qkv.unbind(0)
    y=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=self.attn_drop.p if self.training else 0.,scale=self.scale)
    return self.proj_drop(self.proj(y.transpose(1,2).reshape(b,n,-1)))
class Network(torch.nn.Module):
    def __init__(self,model,config):
        super().__init__();self.original=model;self.config=config;self.static=None;self.graphed=None
        if config['sdpa']:
            for block in model.backbone.blocks:block.attn.forward=types.MethodType(sdpa_forward,block.attn)
        self.dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[config['precision']]
        # Store encoder weights in the selected dtype, avoiding a second FP32 GPU copy.
        if self.dtype!=torch.float32:model.backbone.to(dtype=self.dtype)
    def encoder(self,x):
        if not self.config['graph']:return self.original.backbone(x)
        if self.graphed is None:
            self.static=torch.empty_like(x,device='cuda').contiguous();self.static.copy_(x)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.original.backbone(self.static)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.graphed=self.original.backbone(self.static)
        self.static.copy_(x);self.graph.replay();return self.graphed
    def forward(self,x):
        x=self.original.normalize(x)[:,:, :,32:-32].to(dtype=self.dtype).contiguous()
        feats=self.encoder(x)
        # Decoder, rotation orthonormalization, camera and body geometry stay FP32.
        return self.original.head(feats.float())

def install_cached_fk():
    """Keep identical matrix operations; remove CUDA scalar reads for static parents."""
    import importlib,weakref
    module=importlib.import_module('models.modules.flame.lbs')
    path=ROOT/'engine/models/modules/flame/lbs.py';tree=ast.parse(path.read_text(encoding='utf8'))
    fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='batch_rigid_transform'))
    cache={}
    def parent_list(parents):
        key=id(parents);entry=cache.get(key)
        if entry is None or entry[0]() is not parents or entry[1]!=parents._version:
            if len(cache)>=8:cache.clear()
            entry=(weakref.ref(parents),parents._version,parents.detach().cpu().tolist());cache[key]=entry
        return entry[2]
    class Replace(ast.NodeTransformer):
        def visit_Subscript(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='transform_chain' and isinstance(node.slice,ast.Subscript) and isinstance(node.slice.value,ast.Name) and node.slice.value.id=='parents':
                node.slice.value.id='static_parents'
            return self.generic_visit(node)
    fn=Replace().visit(fn);fn.body.insert(0,ast.parse('static_parents = _parent_list(parents)').body[0])
    scope=dict(module.__dict__);scope['_parent_list']=parent_list
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<cached static EHM FK parents>','exec'),scope)
    module.batch_rigid_transform=scope['batch_rigid_transform']

def trim_unused_geometry(ehm):
    # These constructor/training buffers are not read by EHM_v2.forward.
    for model in [ehm,ehm.smplx]:
        for name in ['laplacian_matrix','laplacian_matrix_negate_diag','head_idxs_uv_flat','head_idxs_temp']:
            if hasattr(model,name):delattr(model,name)
