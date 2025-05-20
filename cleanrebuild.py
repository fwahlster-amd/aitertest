import json
import os
import shutil
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
jit_dir = f"{this_dir}/aiter/aiter/jit"
csrc_dir = f"{this_dir}/aiter/csrc"
build_dir = f"{this_dir}/aiter/aiter/jit/build"

CLEAN_ALL: bool = False
REBUILD_ALL: bool = False
REBUILD_INPLACE: bool = True
DEBUG_CONFIG: bool = True

def delete_libs(jit_folder: str):
    if os.path.exists(jit_folder) and os.path.isdir(jit_folder):
        for file in os.listdir(jit_folder):
            if file.endswith(".so"):
                so_path = os.path.join(jit_folder, file)
                print(f"removing {so_path}")
                try:
                    os.remove(so_path)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

#os.environ['AITER_REBUILD'] = '1'
#AITER_REBUILD = int(os.environ.get("AITER_REBUILD", "0"))

# JIT_WORKSPACE_DIR need to be set before first call to get_user_jit_dir!
#os.environ['JIT_WORKSPACE_DIR'] = os.path.join(this_dir, "jit_workspace")

jit_workpace_fir = os.environ.get('JIT_WORKSPACE_DIR', None)

sys.path.insert(0, f"{this_dir}/aiter/aiter")
from jit import core

user_jit_dir = core.get_user_jit_dir()
print(f"user_jit_dir: {user_jit_dir}")

if CLEAN_ALL:
    # delete .so files in /aiter/aiter/jit
    delete_libs(jit_folder=jit_dir)

    if os.path.exists(user_jit_dir) and os.path.isdir(user_jit_dir) and user_jit_dir != jit_dir:
        shutil.rmtree(user_jit_dir)

    if os.path.exists(build_dir) and os.path.isdir(build_dir):
        shutil.rmtree(build_dir)
    
os.makedirs(user_jit_dir, exist_ok=True)
os.makedirs(build_dir, exist_ok=True)

build_op_names: list = ['module_norm']

if REBUILD_ALL:
    with open( os.path.join(jit_dir,"optCompilerConfig.json" ), "r") as file:
        data: dict = json.load(file)
        build_op_names = data.keys()

for ops_name in build_op_names:
    args = core.get_args_of_build(ops_name=ops_name)
    is_python_module: bool = args["is_python_module"]
    is_standalone: bool = args["is_standalone"]
    torch_exclude: bool = args["torch_exclude"]
    if not is_python_module:
        continue

    # replace -O3 with O0
    flags_extra_cc: list = args["flags_extra_cc"]
    flags_extra_hip: list = args["flags_extra_hip"]

    if DEBUG_CONFIG:
        if '-O3' not in flags_extra_cc and '-O3' not in flags_extra_hip:
            flags_extra_cc += ["-O0", "-ggdb3"]
            flags_extra_hip += ["-O0", "-ggdb3"]

    if REBUILD_INPLACE:
        core.rm_module(md_name=ops_name)
        core.clear_build(md_name=ops_name)

    core.build_module(
        md_name=ops_name, # module_norm
        srcs=args["srcs"], # + [csrc_dir]
        flags_extra_cc=flags_extra_cc,
        flags_extra_hip=flags_extra_hip,
        blob_gen_cmd=args["blob_gen_cmd"],
        extra_include=args["extra_include"],
        extra_ldflags=args["extra_ldflags"],
        verbose=True,
        is_python_module=True,
        is_standalone=is_standalone,
        torch_exclude=torch_exclude,
    )

    print(f"Builing {ops_name} done!")

# run test on debug modules
sys.path.insert(0, f"{this_dir}/aiter/op_tests")
from op_tests import test_layernorm2d
from utility import dtypes

test_layernorm2d.test_layernorm2d_fuseAdd(dtypes.bf16, 128, 8192)