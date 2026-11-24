# Unity streaming (experimental)

Streams **prepared offline motion** while playing or scrubbing RGB MoCap. Includes
body rotations, available finger rotations, and pelvis translation/root motion for
multiple selected people. No inference is launched by streaming. PEAR live capture,
face streaming, mesh transfer and recording Unity AnimationClips are not included.

## Install and connect

1. In Unity 6, open **Window > Package Management > Package Manager**, choose
   **Install package from disk**, and select
   `unity_plugin/RGBMoCap/package.json` in the RGB MoCap installation.
   Alternatively copy the `RGBMoCap` folder into your Unity project's `Assets`.
2. Add **GameObject > RGB MoCap > Stream Receiver**. One receiver handles all people.
3. Add **RGB MoCap > Stream Actor** to each character's top-level object.
   Assign the receiver and its Animator. Use a valid Humanoid Avatar for automatic
   body/finger mapping, or fill the `Mapping` overrides for a generic skeleton.
4. Characters must start in a reference pose matching the source, normally T-pose,
   facing Unity +Z, at metre scale. An arbitrary A-pose is not automatically corrected.
   Do not run other IK/retarget components on the same bones simultaneously.
5. In RGB MoCap choose **Stream > Unity**, select people, then play or scrub.
6. Enter Unity **Play Mode**. Select a person in each Stream Actor's inspector.
   A unique name such as `Person 1` also works; the dropdown stores a persistent ID
   so duplicate names cannot silently drive the wrong character.

Both applications must run on the **same PC** with the same port (default 49168).
There is no Unreal SDK, native build or external Unity package dependency.
The receiver reconnects automatically. Change the port before entering Play Mode.
Enable Scene view Gizmos to inspect all received skeletons without a mesh.

## Root motion and proportions

Only rotations are transferred for body/fingers. Target bone lengths and local
translations stay intact; pelvis translation is the exception. Enable **Root Motion**
to apply source pelvis displacement relative to its reference, including actor scene
position and vertical motion. The character's initial placement is an additive offset.
`Motion Scale` scales translation only (1 = source metres), never bone sizes. It can
compensate for a taller/shorter target. Source world facing/root rotation is carried
by the pelvis rotation; Unity Animator root-motion extraction is not used.
This is a reference-relative rotation retarget, not a contact/foot-IK solver;
different proportions can require a downstream contact correction solution.

The component disables its Animator while active and restores it and the original
local pose when disabled. On disconnect/missing frames it holds the last pose.
Do not disable/re-enable the component in an arbitrary animated pose to recalibrate.
Mapping edits during playback require **Rebuild bone mapping**. Missing finger bones
are skipped; a source without tracked finger motion cannot synthesize it.

## Protocol / maintenance

Loopback TCP, one receiver connection. Four-byte unsigned little-endian payload
length followed by UTF-8 JSON. Zero length = heartbeat. Version 1 snapshots contain
all currently active actors with persistent ID/name/rig key, source frame, bone names,
parents and flattened global 3x4 reference/pose matrices. Unity X-right/Y-up/Z-forward,
metres, row-major. Blender Z-up is converted using `(x,y,z) -> (x,z,y)` on translation
and conjugation on rotations. Maximum 128 people, 256 bones/person, 8 MiB/frame.
These are protocol limits, not a realtime performance guarantee.
The producer and receiver each retain only their latest pending snapshot.
Rig/schema travels with every frame so reconnection and dropped pending frames are
self-contained. Networking runs on a background thread; Unity objects are accessed
on the main thread only. This initial implementation prioritises compatibility over
compact binary bandwidth. No firewall opening or network exposure is needed.
