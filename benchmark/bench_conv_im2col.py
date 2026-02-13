import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyperf
from benchmark.scripts.conv_im2col_one_pass import main as run_test

def bench_im2col():
    run_test()

if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_func("atc_im2col", bench_im2col)

