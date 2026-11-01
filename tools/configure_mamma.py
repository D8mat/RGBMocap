"""Register an existing, separately validated Windows research environment."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap.mamma_runtime import configure

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--reviewed-mamma-license', action='store_true', required=True)
    args = parser.parse_args()
    print(configure(args.python, args.engine))
