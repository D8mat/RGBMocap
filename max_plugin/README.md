# RGB MoCap for 3ds Max 2025

## Install and open

In Max, use Scripting > Run Script and choose `install.py` in this folder. It copies the plugin to your user scripts and registers the **RGB MoCap / RGBMoCapFBXLiveLink** action, which can be assigned to a toolbar/menu through Customize User Interface. To open without installation, run `rgbmocap_max.py` directly. Requires Max 2025's bundled Python and PySide6; no pip packages or separate Chamiseul receiver.

## Connect and map

1. In RGB MoCap, prepare motion, select **Stream > 3ds Max** and check people. The application listens on local port **49169**.
2. In Max open the plugin, then **Connect**. Import a conventional skinned FBX character in its native reference pose. Disable Auto Key.
3. Choose a source person in the plugin. Select exactly one skeleton root in Max, click **Use Selected Skeleton / Auto Map**, review the source/target bone table, then **Bind Mapping**. Names match ignoring a namespace. Use dropdowns for different names (for example Hips to pelvis/Reference). At least the source Hips/root must be mapped. Bind each person to a different skeleton. No Biped, CAT, constraints, mirrored/sheared rigs or procedural controllers.
4. Source and target reference poses should correspond. This first version does not automatically solve A-pose versus T-pose or foot IK. It transfers rest-relative world rotations, preserving target bone lengths, and root movement in scene units. Root travel scale is independent of character scale. The character remains relative to its original placement in Max.

## Live, Record, Bake

- **Live Preview** temporarily applies incoming poses. Turning it off or disconnecting restores pre-preview animation. No keys are written.
- **Record** records incoming samples for all bound actors on Max's timeline, starting at Target start frame. Time uses the Max frame rate and elapsed real time. **Stop / Keep Record** retains the keys; **Cancel / Restore** discards the unfinished recording. Live recording is sampled: if playback/network/UI skips a pose it cannot reconstruct it. Use Bake for complete offline motion.
- **Bake Selected Person** requests the full prepared timeline from RGB MoCap, independently of playback. Source range is zero-based; the default end is clamped naturally by the available frames. Motion duration is preserved when the source and Max frame rates differ, using subframe keys. Target start frame offsets the result. Takes with no visible source person leave gaps. A successful bake unchecks that source person in RGB MoCap, so streaming does not overwrite the baked animation.
- **Cancel / Restore**, disconnects and errors restore the target controllers captured before the unfinished operation. Keys outside the written range are retained. Existing keys at written times may be replaced; save your scene before recording/baking. Each operation copies the mapped controllers for recovery, which needs additional RAM on large animations. Closing the panel cancels unfinished work.

To remove the plugin, delete its user-script folder and RGBMoCap-FBXLiveLink.mcr from user macros. It does not alter other Max plugins, scene files, or the source mocap project. The panel must be rebound after reopening; mappings are session-local in this version.
