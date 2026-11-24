# Upstream integration — 2026-11-24

Integrated toyxyz/toyxyz_RGBMocap from 7c5fea2a24dd5e33ea102ee05fe72470b7bc52f3 through d272f18e256066be555812df293edf9c5d26b657.

- Moving-camera FlowHMR shares validated original-frame VGGT intrinsics across scene takes, SAM features and placement. Feature receipts include those intrinsics; inference rejects stale or mismatched feature calibration. Static-camera behavior remains independent of VGGT.
- Adopted upstream explanatory comments for the UE sender Latest build settings. The executable settings already matched our UE 5.8 fix; no runtime rebuild was performed.
- Updated upstream Live Link and validation documentation. Those upstream engine validation statements describe the author's tests, not new local tests.
- Preserved local retarget reference-pose selection, character positioning, Unity streaming, face preview/overlay, viewport features, custom exports and 5000-frame limits. Origin is unchanged.

Validation: 59 tests passed in the isolated source staging tree, including all five new FlowHMR intrinsics tests, export/default settings, model-manager fixtures, Unreal stream/reference and mannequin retarget tests. Existing model paths were used read-only for receipt metadata. No full GPU reconstruction, Unreal rebuild or real-time streaming test was performed in this update. No models, runtime packages or user projects were modified.

Existing moving-camera FlowHMR feature caches may need Reconstruct 3D to refresh them. Missing/incomplete camera estimates require camera estimation first. Existing exported results were not rewritten.
