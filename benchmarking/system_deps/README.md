# Benchmark System Dependencies

Each subdirectory defines one benchmark dependency group. Benchmark YAML entries
declare groups with `dependencies:`, and `curator-benchmark` runs the matching
scripts when system dependency setup or checking is requested.

The convention is:

```text
benchmarking/system_deps/<dependency-group>/check.sh
benchmarking/system_deps/<dependency-group>/install.sh
```

`check.sh` should exit `0` when the dependency is available and nonzero when it
is missing or unusable. `install.sh` should be idempotent and should fail
clearly when the current environment cannot be modified.

System dependency scripts should be self-contained within the benchmark suite.
They should not depend on caller-provided environment variables or files outside
the `benchmarking/` directory, because container workflows may mount only the
benchmark suite package and use the Curator installed in the image as the
Curator package under test.

If a script needs helper files, place them under the dependency-group directory
or another benchmark-owned path and derive paths from the script location:

```bash
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
"${script_dir}/helper.sh"
```

This works when `curator-benchmark` runs the script from a mounted benchmark
suite in a Docker container, when setup runs on bare metal, and when a developer
invokes the script manually. It also avoids coupling system dependency scripts
to the Curator package under test, which may be a released package or image
rather than the checkout that owns the benchmark suite.
