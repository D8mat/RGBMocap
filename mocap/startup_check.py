"""Opt-in smoke receipt for standalone startup and inherited worker paths."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer
from .paths import ROOT, HOME, MODELS, PROJECTS, worker_command
from .storage import atomic_json

KEYS = ("RGB_MOCAP_DATA", "PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
        "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME")

def schedule_check(window, report, finished):
    process = QProcess(window)
    timer = QTimer(window)
    timer.setSingleShot(True)
    done = False
    code = (
        "import os,sys,json;from pathlib import Path;"
        "sys.path.insert(0,sys.argv[1]);"
        "from mocap.paths import ROOT,HOME,MODELS,PROJECTS;"
        "print(json.dumps(dict(root=str(ROOT),home=str(HOME),models=str(MODELS),"
        "projects=str(PROJECTS),python=sys.executable,cwd=os.getcwd(),"
        "environment={k:os.environ.get(k) for k in " + repr(KEYS) + "})))"
    )
    def complete(success, message=""):
        nonlocal done
        if done:
            return
        done = True
        timer.stop()
        output = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        error = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        worker = None
        try:
            if success:
                worker = json.loads(output)
        except (ValueError, TypeError) as exc:
            success, message = False, str(exc)
        atomic_json(Path(report), dict(
            success=success, message=message, frozen=bool(getattr(sys, "frozen", False)),
            root=str(ROOT), home=str(HOME), models=str(MODELS), projects=str(PROJECTS),
            cwd=os.getcwd(), environment={k:os.environ.get(k) for k in KEYS},
            inference_imported="torch" in sys.modules, worker=worker, stderr=error))
        process.deleteLater()
        timer.deleteLater()
        finished(success)
    def timeout():
        process.kill()
        complete(False, "Startup worker check timed out")
    process.setWorkingDirectory(str(ROOT))
    environment = QProcessEnvironment.systemEnvironment()
    environment.insert("PYTHONIOENCODING", "utf-8")
    process.setProcessEnvironment(environment)
    process.finished.connect(lambda code, status: complete(
        code == 0 and status == QProcess.NormalExit, f"Worker exit code {code}"))
    process.errorOccurred.connect(lambda error: complete(False, process.errorString()))
    timer.timeout.connect(timeout)
    process.start(worker_command("diagnose")[0], ["-B", "-c", code, str(ROOT)])
    timer.start(10000)

