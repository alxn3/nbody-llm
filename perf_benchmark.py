#!/usr/bin/env python3

# SBATCH --job-name=benchmark_nbody
# SBATCH --nodes=1
# SBATCH --exclude=storage-[1-8]

import argparse
import os
import sys
import subprocess

parser = argparse.ArgumentParser("Benchmarking tool")
parser.add_argument("executable", help="Executable to benchmark")
args = parser.parse_args()

# check if executables exist
executable = args.executable
if not os.path.exists(executable):
    print(f"Executable {executable} does not exist")
    sys.exit(1)

base_name = os.path.basename(executable)

threads = []
x = os.cpu_count()
while x > 0:
    threads.append(x)
    x = x // 2
points = [10, 30, 50, 75, 100, 300, 500, 750, 1000, 3000, 5000, 7500, 10000, 30000, 50000, 75000, 100000]
bf_max = 3000
thread_max = {
    1: 10000,
    2: 30000,
    4: 50000,
}

output_csv = f"perf_benchmark_{base_name}.csv"

env = os.environ.copy()

DELIM = ";"

# flags = "fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.scalar_single,fp_assist.any"

# fp_ret_sse_avx_ops.all
#        [All FLOPS]
#   fp_ret_sse_avx_ops.dp_add_sub_flops
#        [Double precision add/subtract FLOPS]
#   fp_ret_sse_avx_ops.dp_div_flops
#        [Double precision divide/square root FLOPS]
#   fp_ret_sse_avx_ops.dp_mult_add_flops
#        [Double precision multiply-add FLOPS. Multiply-add counts as 2 FLOPS]
#   fp_ret_sse_avx_ops.dp_mult_flops
#        [Double precision multiply FLOPS]
#   fp_ret_sse_avx_ops.sp_add_sub_flops
#        [Single-precision add/subtract FLOPS]
#   fp_ret_sse_avx_ops.sp_div_flops
#        [Single-precision divide/square root FLOPS]
#   fp_ret_sse_avx_ops.sp_mult_add_flops
#        [Single precision multiply-add FLOPS. Multiply-add counts as 2 FLOPS]
#   fp_ret_sse_avx_ops.sp_mult_flops
#        [Single-precision multiply FLOPS]

flags = "fp_ret_sse_avx_ops.all,fp_ret_sse_avx_ops.dp_add_sub_flops,fp_ret_sse_avx_ops.dp_div_flops,fp_ret_sse_avx_ops.dp_mult_add_flops,fp_ret_sse_avx_ops.dp_mult_flops,fp_ret_sse_avx_ops.sp_add_sub_flops,fp_ret_sse_avx_ops.sp_div_flops,fp_ret_sse_avx_ops.sp_mult_add_flops,fp_ret_sse_avx_ops.sp_mult_flops"

i = 0
with open(output_csv, "w") as f:
    for thread in threads:
        for point in points:
            if point > bf_max and "bf" in executable:
                continue
            if thread in thread_max and point > thread_max[thread]:
                continue
            cmd = [executable, "-t", str(thread), "-n", str(point)]
            perf_all = ["perf", "stat", "-x;", "-ddd", "-r", "3"]
            perf_flop = ["perf", "stat", "-x;", "-e", flags, "-r", "3"]
            print(perf_all + cmd)
            pall = subprocess.run(perf_all + cmd, stderr=subprocess.PIPE, env=env)
            pflop = subprocess.run(perf_flop + cmd, stderr=subprocess.PIPE, env=env)
            lines = pall.stderr.decode().strip().split(
                "\n"
            ) + pflop.stderr.decode().strip().split("\n")
            f.write(
                "\n".join(
                    [
                        f'"{executable}"{DELIM}{thread}{DELIM}{point}{DELIM}{x}'
                        for x in lines
                    ]
                )
            )
            f.write("\n")
            if i % 10 == 0:
                print("\n".join(
                    [
                        f'"{executable}"{DELIM}{thread}{DELIM}{point}{DELIM}{x}'
                        for x in lines
                    ]
                ), file=sys.stderr)
                print(f"Progress: {i}/{len(threads) * len(points)}", file=sys.stderr)
            i += 1
