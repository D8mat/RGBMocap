# Source package validation

Prepared on Windows on 27 October 2026. Application modules are copied from the
current app, including character retargeting, FBX rest-pose handling, VMD export
and the latest viewport/name/checkbox changes. Source preparation does not change
inference, motion correction or retargeting calculations.

Composition checks cover Python syntax, local documentation links, exact
application-source identity, five separate AI dependency profiles, deterministic
local wheel reconstruction, auxiliary-data hashes and Git exclusion rules.
Registered models, user videos, characters, motion, projects, experimental scripts,
virtual environments, installers and AI instruction documents are excluded.
The explicitly listed upstream shader libraries/toon ramps remain as licensed
editable importer resources.

See [Build from source](BUILD.md) for the build and release validation procedure.
A fresh GUI EXE and Windows Setup were built using this checkout's source and
the existing development dependencies. The native correction module was reused
from the reviewed build; its source matches the current checkout. The complete
Blender 4.5 character runtime and rebuilt OpenCC wheel were staged separately.
The GUI startup check and VRM/PMX/OpenCC add-on loading passed. All 18 installer
engine regression cases passed, covering download recovery, cancellation and
transactional repair. Application payload hashes and installer sizing were also
checked. No previous application or installer was used as the build input.

A build check is not a fresh-PC installation or a new inference/retargeting test
for every model and character. Hardware compatibility remains limited to the
configurations actually tested in the [user guide](USER_GUIDE.md).

## Version 1.0.1 packaging fix

Version 1.0.0 omitted historical requirements files used by worker cache
fingerprints. This could stop reconstruction after Find People with a
`requirements-windows.txt` FileNotFoundError. Version 1.0.1 includes those
unchanged files and rejects incomplete source/staged applications before
building Setup. Model calculations and managed dependency pins are unchanged.
