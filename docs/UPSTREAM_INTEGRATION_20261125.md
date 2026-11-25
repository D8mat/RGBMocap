# Upstream integration — 2026-11-25

Integrated toyxyz/toyxyz_RGBMocap from d272f18 to a7fde78ed49b21f51968dc3190537c41c8f6833f, preserving the local fork and origin.

Added upstream CLI for offline/live workflows with shared desktop orchestration, single-pass timeline playback (Loop toggle), optional default-enabled VGGT-Omega camera optimization with BF16/SDPA and native-BF16 capability fallback, Windows camera-progress file sharing fix, UE 5.6+ setup/prebuilt bundle support and updated guides/sample videos.

Local merge resolutions preserve 5000-frame limits in desktop and CLI; MAMMA fit-cache code identity; custom Unreal export settings and grounding/heading behavior; Blender reference pose; single-person scene placement; Track Face preview; Unity/Max streaming; Manny/Quinn/Puppet import and all existing viewport/retarget controls. Shared export_row now carries the local options for both frontends. No project/model/runtime data was migrated or downloaded.

Validation before deployment: 85 tests passed (CLI, playback, defaults, model management fixtures, FlowHMR intrinsics, stream/reference and mannequin retarget). Custom GUI, six-model export dialogs, sidebar/theme, Unity, Track Face, reference-pose selection, Move Character, camera planes/orthographic, RIFE gating and own Max target passed. Targeted shared-export tests covered custom Unreal options, Blender A-pose and scene transforms. Changed Python sources parsed successfully. Existing Max plugin sources are byte-identical.

Actual Unreal 5.7 sender startup and local provider handshake passed using the existing installed bridge with the updated Python sender. No provider rebuild or new receiver download was needed. No receiving Unreal project session, new full GPU motion/camera reconstruction, or prebuilt ZIP installation was tested. VGGT memory/quality claims in upstream documentation are the author's measurements.
