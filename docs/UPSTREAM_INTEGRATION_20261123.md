# Upstream integration — 2026-11-23

Integrated toyxyz/toyxyz_RGBMocap from 5f57918 through 7c5fea2.
The local fork remote remains unchanged; no push performed.

- Unified workflow model management, installed-file details and confirmed removal.
- Remember export choices and the last offline AI model; Defaults in export dialogs.
- Save/reset/confirm starting settings for new clips.
- Live appearance settings (theme and UI size).
- Optional RIFE 4.9 frame interpolation before reconstruction for low-FPS sources.
- Native Blender FBX importer for legacy/ASCII FBX and specular texture preservation.
- Windows Smart App Control and long-install-path recovery documentation.

Preserved local features: Unreal export and UE5.8 bridge build fix, Unity streaming,
Track Face-backed preview/overlay, 5,000-frame model limits, MAMMA fit cache,
Toyxyz mapping and virtual target A/T reference, manual placement, camera planes,
playblast/material colours, orthographic/multiple viewports, checker floor and
expandable/collapsible sidebar. Appearance resizing does not restore the upstream
330-pixel sidebar cap or expand collapsed panels. Viewport text remains white.

Validation: 54 upstream tests passed (local export request expectations extended
for our existing Unreal options). Offscreen custom GUI and six export profiles,
appearance/sidebar collapse, Unity switching and RIFE gating checks passed.
Blender 4.5 native importer -> Puppet V5 -> 20-frame Toyxyz bake -> skinned FBX
passed with Hips -> Reference preserved. No real RIFE/GPU inference, model removal,
clean installation or Unreal receiver session was run. No model data downloaded,
removed, or source-project data rewritten.
