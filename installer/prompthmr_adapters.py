"""Small Windows adaptations applied to user-downloaded PromptHMR source.

Restricted upstream sources are not bundled with RGB MoCap. The replacements
match the pinned release and fail closed if it changes.
"""
from .mamma_adapters import replace


def adapt_engine(root):
    replace(root/'pipeline/camera/__init__.py',
        'from .masked_droid_slam import run_metric_slam, calibrate_intrinsics, run_slam',
        '\n'.join('def '+name+'(*args, **kwargs):\n    raise RuntimeError("Moving-camera PromptHMR is not supported by this Windows adapter.")\n' for name in ('run_metric_slam','calibrate_intrinsics','run_slam')))
    replace(root/'pipeline/detector/sam2_video_predictor.py', ', load_video_frames_from_np', '')
    replace(root/'pipeline/tools.py', 'num_workers=4 if os.cpu_count() > 4 else os.cpu_count()', 'num_workers=0')
    replace(root/'pipeline/tools.py', "model_cfg = '/' + os.path.abspath(model_cfg)", "model_cfg = os.path.abspath(model_cfg).replace('\\\\', '/')")
    replace(root/'pipeline/detector/sam2_video_predictor.py', 'cfg = compose(config_name=config_file, overrides=hydra_overrides)',
        "cfg = OmegaConf.merge(OmegaConf.load(config_file), OmegaConf.from_dotlist([x.lstrip('+') for x in hydra_overrides]))")
    replace(root/'prompt_hmr/models/components/clip_encoder.py', "pretrained='metaclip_fullcc'", 'pretrained=None')
    replace(root/'pipeline/phmr_vid.py', 'batch_size=16', 'batch_size=1')
    replace(root/'pipeline/spec/cam_calib.py', 'torch.load(CKPT)', 'torch.load(CKPT, weights_only=False)')
    replace(root/'pipeline/gvhmr/hmr4d/model/gvhmr/gvhmr_pl_demo.py', 'torch.load(ckpt_path, "cpu")', 'torch.load(ckpt_path, "cpu", weights_only=False)')
    replace(root/'prompt_hmr/models/components/twoway_transformer.py', 'from xformers.ops import memory_efficient_attention',
        '''def memory_efficient_attention(q, k, v, attn_bias=None):
    if attn_bias is not None:
        raise ValueError('Unsupported attention bias')
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), dropout_p=0.).transpose(1,2)
''')


def install_python_source(source, target, name):
    """These selected Detectron2 operators use the pinned torchvision kernels."""
    import shutil
    if name=='detectron2':
        shutil.copytree(source/'detectron2',target/'detectron2')
        # Preserve the provider license beside the package.
        shutil.copy2(source/'LICENSE',target/'detectron2/LICENSE')
    else:
        raise ValueError('Unknown Python source package')
