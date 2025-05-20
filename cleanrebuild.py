import json
import os
import shutil
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
jit_dir = f"{this_dir}/aiter/aiter/jit"
csrc_dir = f"{this_dir}/aiter/csrc"
build_dir = f"{this_dir}/aiter/aiter/jit/build"

REBUILD_ALL: bool = False

if REBUILD_ALL:
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

os.environ['AITER_REBUILD'] = '1' # AITER_LOG_MORE
AITER_REBUILD = int(os.environ.get("AITER_REBUILD", "0"))

# JIT_WORKSPACE_DIR need to be set before first call to get_user_jit_dir!
os.environ['JIT_WORKSPACE_DIR'] = os.path.join(this_dir, "jit_workspace")
sys.path.insert(0, f"{this_dir}/aiter/aiter")
from jit import core
# optCompilerConfig.json
user_jit_dir = core.get_user_jit_dir()
print(f"user_jit_dir: {user_jit_dir}")

if REBUILD_ALL:
    if os.path.exists(user_jit_dir) and os.path.isdir(user_jit_dir):
        shutil.rmtree(user_jit_dir)

with open(this_dir + "/optCompilerConfig.json", "r") as file:
    data: dict = json.load(file)
    for ops_name, vals in data.items():
        args = core.get_args_of_build(ops_name=ops_name)
        #is_python_module: bool = vals.get("is_python_module", default=False)
        is_python_module: bool = args["is_python_module"]
        if not is_python_module:
            continue

        # remove pybind, because there are already duplicates in rocm_opt
        #new_list = [el for el in all_opts_args_build["srcs"] if "pybind.cu" not in el]
        # replace -O3 with O0
        flags_extra_cc: list = args["flags_extra_cc"]
        flags_extra_hip: list = args["flags_extra_hip"]

        if '-O3' not in flags_extra_cc and '-O3' not in flags_extra_hip:
            flags_extra_cc += ["-O0", "-ggdb3"]
            flags_extra_hip += ["-O0", "-ggdb3"]

        core.build_module(
            md_name=ops_name, # module_norm
            srcs=args["srcs"] + [csrc_dir],
            flags_extra_cc=flags_extra_cc,
            flags_extra_hip=flags_extra_hip,
            blob_gen_cmd=args["blob_gen_cmd"],
            extra_include=args["extra_include"],
            extra_ldflags=None,
            verbose=True,
            is_python_module=True,
            is_standalone=False,
            torch_exclude=False,
        )