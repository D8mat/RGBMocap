# AI model guides

Choose a model in **AI Motion Model** before preparing its files or opening
videos. ExMR is the default for new work; reopening a project restores its
saved selection. Only implemented tools appear in the list.

| Model | Input and body | Fingers | Detailed guide |
| --- | --- | --- | --- |
| ExMR | One video; SMPL-X | Included | [ExMR](EXMR.md) |
| FlowHMR | One video; SMPL-H | Included | [FlowHMR](FLOWHMR.md) |
| GVHMR | One video; SMPL-X | Optional HandFlow | [GVHMR](GVHMR.md) |
| MAMMA | Two or more synchronized fixed-camera videos; SMPL-X | Included | [MAMMA](../MAMMA.md) |

One-video input does not mean one-person output. ExMR, FlowHMR and GVHMR can
process checked people independently after **Find person**. Without discovery,
automatic tracking selects one person and may switch identity during overlap.
MAMMA detects and matches people across views during reconstruction; its
**Find person** button is hidden.
For manual MAMMA cameras, see the [calibration format and JSON example](../MAMMA.md#manual-cameras).

In this project's subjective tests, ExMR was generally fast with good quality;
FlowHMR was slower with the best overall quality; GVHMR was less consistent
overall but sometimes better for unusual poses such as backflips. These are
not benchmark rankings. MAMMA is a separate multi-view workflow, not another
entry in that single-video ranking.

## Installation and shared workflow

1. Run `install.cmd` in the app folder. It prepares the runtimes for all three
   single-video models and their supported add-ons, Blender, FFmpeg and native
   motion correction. It does **not** download AI weights or registered body data.
2. For MAMMA, additionally follow [its runtime setup](../MAMMA.md#install-the-runtime).
   `Setup_MAMMA.cmd` is separate from the base installation and update commands.
3. Run `RGBMoCap.cmd`, select the model and use **Model Manager**. Follow each
   row's exact download/import instructions. Some files require registration
   and license acceptance; Google Drive downloads may need manual import.
4. Open footage, trim, reconstruct and review. Contact correction defaults on;
   Foot defaults on and Hand off. Edit contact timings and **Apply Correction**
   without rerunning AI inference. Export the displayed applied motion as FBX.

Model-specific settings affect reconstruction; they are separate from contact
correction and display-only viewport controls. Original and corrected results
are preserved. One project can hold separate results for each model. MAMMA saves
all of its camera videos with **Save Project As**.

Use **Resource Mode > Automatic** initially. Actual RAM/Windows allocation
headroom monitoring, CUDA allocator budgets and allocation-error reporting remain
active. More frames, pixels, people or views can increase work; no setting
guarantees that every clip fits. Low Memory adjusts supported single-video
batches; it does not select a reduced-view or lower-quality MAMMA preset.
See [resources](../USER_GUIDE.md#model-and-processing-settings)
and [measured validation scope](../VALIDATION.md).

See the [user guide](../USER_GUIDE.md) for shared controls, contacts, retargeting,
FBX/VMD export, storage and cleanup. For agent-assisted setup and diagnostics,
use [the agent guide](../AGENT_GUIDE.md). Model code, weights and body data retain
their own [licenses](../../THIRD_PARTY_NOTICES.md).
