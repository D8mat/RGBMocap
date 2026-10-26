"""Small auditable Windows patches; upstream license files stay intact."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(relative, before, after):
    path = ROOT / "third_party" / relative
    text = path.read_text(encoding="utf-8")
    if after in text:
        return
    if before not in text:
        raise RuntimeError(f"Upstream changed; inspect patch: {relative}")
    path.write_text(text.replace(before, after), encoding="utf-8")


def main():
    replace('GVHMR/hmr4d/network/gvhmr/relative_transformer.py',
            '        if L > self.max_len:\n',
            '        if L > self.max_len and getattr(self, "_mocap_split_queries", False):\n'
            '            attnmask = self.max_len  # Local Windows full-key query splitting.\n'
            '        elif L > self.max_len:\n')
    replace("GVHMR/hmr4d/utils/preproc/__init__.py",
            "from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO",
            "# Windows static-camera inference does not import optional SimpleVO.")
    replace("GVHMR/hmr4d/utils/geo_transform.py", "import pytorch3d.ops.knn as knn",
            "from pytorch3d.ops import knn  # Windows dense exact fallback")
    # Training-only visualization must not import the compiled renderer at startup.
    replace("GVHMR/hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py",
            "from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines",
            "# Training-only wis3d visualization is excluded from the Windows inference path.")
    replace("GVHMR/hmr4d/model/gvhmr/utils/postprocess.py",
            "from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines",
            "# Unused development visualization is excluded from Windows inference.")
    replace("HTD-Refine/utils/vis.py", "from utils.renderer import Renderer", "# Windows GUI supplies rendering; no compiled PyTorch3D renderer is loaded.")
    replace("HTD-Refine/optimization/scheduler.py",
            "super().__init__(optimizer, last_epoch, verbose)",
            "super().__init__(optimizer, last_epoch=last_epoch)  # PyTorch 2.7 removed verbose.")
    # Explicit model loading is needed after PyTorch's weights_only default change.
    hmr_path=ROOT/'third_party/GVHMR/hmr4d/network/hmr2/__init__.py'
    mapped='torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)'
    if mapped not in hmr_path.read_text(encoding='utf-8'):
        replace("GVHMR/hmr4d/network/hmr2/__init__.py", 'torch.load(checkpoint_path, map_location="cpu")',
                'torch.load(checkpoint_path, map_location="cpu", weights_only=False)')
    replace("GVHMR/hmr4d/network/hmr2/__init__.py", 'torch.load(checkpoint_path, map_location="cpu", weights_only=False)',mapped)
    replace('GVHMR/hmr4d/network/hmr2/__init__.py','    model = HMR2(model_cfg)',
            "    # Local adapter: initialize shapes only and map the existing checkpoint weights.\n    with torch.device('meta'):\n        model = HMR2(model_cfg)")
    replace('GVHMR/hmr4d/network/hmr2/__init__.py','model.load_state_dict(state_dict, strict=True)',
            'model.load_state_dict(state_dict, strict=True, assign=True)')
    replace('GVHMR/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py',
            'def build_model(model_name, checkpoint=None):\n',
            "def build_model(model_name, checkpoint=None):\n"
            "    # Local adapter: avoid constructing a second multi-GB CPU weight copy.\n"
            "    if checkpoint is not None:\n"
            "        with torch.device('meta'):\n"
            "            pose = build_model(model_name)\n"
            "        check = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)\n"
            "        pose.load_state_dict(check.get('state_dict', check), strict=True, assign=True)\n"
            "        return pose\n")
    for file in ('GVHMR/hmr4d/network/hmr2/vit.py','GVHMR/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/builder/backbones/vit.py'):
        replace(file,'torch.linspace(0, drop_path_rate, depth)',"torch.linspace(0, drop_path_rate, depth, device='cpu')")
    replace("GVHMR/hmr4d/model/gvhmr/gvhmr_pl_demo.py", "torch.load(ckpt_path, \"cpu\")",
            "torch.load(ckpt_path, \"cpu\", weights_only=False)")
    if (ROOT / "third_party/FootMR").exists():
        replace('FootMR/hmr4d/utils/preproc/vitpose.py','batch_size = 16',"batch_size = getattr(self, 'batch_size', 16)")
        replace("FootMR/hmr4d/utils/preproc/__init__.py",
                "from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO",
                "# This adapter reuses the existing camera; optional SimpleVO is not imported.")
        replace("FootMR/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py",
                "check = torch.load(checkpoint)", "check = torch.load(checkpoint, map_location='cpu', weights_only=False)")
        replace("FootMR/hmr4d/utils/geo_transform.py", "import pytorch3d.ops.knn as knn",
                "from pytorch3d.ops import knn  # Windows exact fallback")
        replace("FootMR/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py",
                "def build_model(model_name, checkpoint=None):\n",
                "def build_model(model_name, checkpoint=None):\n"
                "    # Avoid allocating/initializing a second 2.5GB CPU model.\n"
                "    if checkpoint is not None:\n"
                "        with torch.device('meta'):\n"
                "            pose = build_model(model_name)\n"
                "        check = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)\n"
                "        pose.load_state_dict(check.get('state_dict', check), strict=True, assign=True)\n"
                "        return pose\n")
        replace("FootMR/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/builder/backbones/vit.py",
                "torch.linspace(0, drop_path_rate, depth)",
                "torch.linspace(0, drop_path_rate, depth, device='cpu')")
    print("Windows inference patches applied.")


if __name__ == "__main__":
    main()
