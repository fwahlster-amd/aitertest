import os
import shutil
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
jit_dir = f"{this_dir}/aiter/aiter/jit"
build_dir = f"{this_dir}/aiter/aiter/jit/build"

if os.path.exists(build_dir) and os.path.isdir(build_dir):
    try:
        shutil.rmtree(build_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

# delete .so files in /aiter/aiter/jit
if os.path.exists(jit_dir) and os.path.isdir(jit_dir):
    for file in os.listdir( jit_dir ):
        if file.endswith(".so"):
            so_path = os.path.join(jit_dir, file)
            print(f"removing {so_path}")
            try:
                os.remove(so_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))


sys.path.insert(0, f"{this_dir}/aiter/aiter")
from jit import core

exclude_ops = ["libmha_fwd", "libmha_bwd"]
all_opts_args_build = core.get_args_of_build("all", exclue=exclude_ops)
# remove pybind, because there are already duplicates in rocm_opt
new_list = [el for el in all_opts_args_build["srcs"] if "pybind.cu" not in el]
all_opts_args_build["srcs"] = new_list

# replace -O3 with O0
all_opts_args_build["flags_extra_cc"] = ["-O0", "-ggdb3"]

core.build_module(
    md_name="aiter_",  # module_norm
    srcs=all_opts_args_build["srcs"] + [f"{this_dir}/csrc"],
    flags_extra_cc=all_opts_args_build["flags_extra_cc"]
    + ["-DPREBUILD_KERNELS"],
    flags_extra_hip=all_opts_args_build["flags_extra_hip"]
    + ["-DPREBUILD_KERNELS"],
    blob_gen_cmd=all_opts_args_build["blob_gen_cmd"],
    extra_include=all_opts_args_build["extra_include"],
    extra_ldflags=None,
    verbose=False,
    is_python_module=True,
    is_standalone=False,
    torch_exclude=False,
)