"""Per-process limits for the lightweight preview executable, before NumPy loads."""
import os


def configure_preview_cpu():
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
