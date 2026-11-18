# RGB MoCap Blender 4.5 compatibility patch

Based on Toyxyz Blender Retargeter 0.4.30, GPL-3.0-or-later.
Original: https://github.com/toyxyz/toyxyz-blender-retargeter

Local changes, 2026-11-18:
- Minimum Blender version changed from 5.0 to 4.5.
- Create Action channelbag curves without the Blender 5-only group_name argument;
  assign the channelbag group explicitly, retaining grouped animation channels.

Verified in bundled Blender 4.5.0: registration, automatic mapping of Puppet V5,
live solve, 20-frame Action bake with the add-on's evaluated-pose verification,
and saving a blend file. Other rig families, full interactive reference editing
and long sequences have not been verified by this compatibility check.
