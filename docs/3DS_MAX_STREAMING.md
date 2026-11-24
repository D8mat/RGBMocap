# 3ds Max 2025 — native RGB MoCap plugin

Our own receiver replaces the experimental Chamiseul HTTP adapter. No separate third-party receiver is required. It uses the existing RGB MoCap bone stream protocol and independent local TCP port 49169. Blender, Unreal and Unity remain separate targets.

Open Stream > 3ds Max > Open 3ds Max Plugin Folder. In Max use Scripting > Run Script to open `max_plugin/rgbmocap_max.py`, or run `install.py` to register the persistent RGB MoCap toolbar action. Follow [the complete plugin guide](../max_plugin/README.md).

Supported: ordinary FBX PRS skeletons; rest-relative rotations, root travel, fingers and multiple mapped actors. Live Preview restores original controllers when stopped. Record captures received live samples at Max's frame rate. Bake requests the complete selected person's prepared timeline independently of playback, supports a source range and target offset, and preserves duration across frame rates. Cancellation or a lost connection restores unfinished controller edits. Bake/Record are controlled inside Max; source project data is unchanged.

Biped/CAT, procedural/constraint controllers, automatic A/T-pose alignment, foot IK and persistent mappings are outside this first release. Record is sampled live data; use Bake to avoid playback frame drops. Save the scene before keying existing animation.
