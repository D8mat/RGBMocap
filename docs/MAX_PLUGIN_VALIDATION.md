# Max 2025 plugin validation — 2026-11-24

Verified with installed 3ds Max 2025.3, Python 3.11.8 and bundled PySide6:

- Real FBX import of bundled Manny: 89 skeleton nodes, original skin retained. Reference and translated keyed poses were written; a test .max scene was saved outside the application project.
- Scene-unit conversion (default Max inches), root movement, original-controller restoration, cancelled recording and retained baked keys.
- Real localhost TCP connection from the shared RGB MoCap sender to the Max plugin: live samples and stop/restore, live Record/Keep, complete 24-frame Bake, and restoration of a subsequent cancelled edit. Qt panel constructed and operated inside Max Batch.
- RGB MoCap offscreen GUI: target selection, dedicated port 49169, inheritance of the existing Bake command handling, clean target shutdown.

These checks used Max Batch and automated panel operations, not a full manual interactive production session. Biped/CAT/constraints, automatic reference-pose correction and foot IK are excluded. Multi-actor mapping is supported by separate actor IDs and bindings, but the full network bake test used one actor. Record is intentionally live sampled; Bake transfers all available frames, including correct source timing and gaps.

The Max node transform cache must be invalidated after restoring copied PRS controllers, otherwise immediately rebinding can capture a stale world transform and double root offsets. This fix is included and covered by the scene tests.
