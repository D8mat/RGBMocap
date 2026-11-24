Upstream integrated through 5f57918 (2026-11-20) from toyxyz/toyxyz_RGBMocap.

Includes Unreal Live Link sender/setup sources and Connect receiver plugin, Manny/Quinn FBX with optional skins, face smoothing and per-person neutral calibration, reference-rig export, and saved MAMMA camera preview recovery.

Preserved local Unreal body export, Blender reference poses, Toyxyz retarget/mapping, humanoid import, camera planes/playblast, floor controls and sidebar sizing. Legacy per-take face calibration remains in face_neutral.json; new per-person calibration uses face_neutral_people.json.

Checks: GUI and all six source export dialogs; 14 upstream protocol/reference/mannequin tests; eight real motion frames exported with Manny and Quinn meshes through Blender; MAMMA saved-video path checks; independent legacy/new neutral calibration. Unreal engine connection/receiver build was not run. Optional Unreal sender still needs Setup_UnrealStream.cmd and matching Unreal/C++ tools.
