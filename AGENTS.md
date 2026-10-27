# Agent guide for RGB MoCap

Read [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) before operating or changing this project. It covers installation, model selection, GUI workflows, retargeting, projects, resources, data protection and the main code paths. Consult the linked user and developer guides for detail.

- This is the Windows Git/CMD distribution. Preserve the separate pinned runtimes and model-specific behavior.
- Treat local `models/`, `projects/`, `cache/`, `character_scenes/`, runtime folders, external project folders and registered model data as user-owned. Do not delete, overwrite, upload or migrate them without appropriate user authorization. Ignore rules are not backups.
- Preserve inference and selected-result behavior when working on preview, contacts or export. Review the relevant code and validation records before changing numerical processing.
- Follow the user's request and perform checks appropriate to the change. Do not claim an inference, import, export, GPU, playback or clean-install test that was not actually run.
