from argparse import ArgumentParser
import os
from enum import Enum


class Backend(Enum):
    SERIAL = 0
    OPENMP = 1
    CUDA = 2
    HIP = 3
    DPCPP = 4
    OPENCL = 5
    LAUNCHER = 6

    def from_str(s: str) -> "Backend":
        s = s.lower()
        if s == "serial":
            return Backend.SERIAL
        if s == "openmp":
            return Backend.OPENMP
        if s == "dpcpp":
            return Backend.DPCPP
        if s == "cuda":
            return Backend.CUDA
        if s == "hip":
            return Backend.HIP
        if s == "opencl":
            return Backend.OPENCL
        if s == "launcher":
            return Backend.LAUNCHER

    def to_str(self) -> str:
        if self == Backend.SERIAL:
            return "serial"
        if self == Backend.OPENMP:
            return "openmp"
        if self == Backend.DPCPP:
            return "dpcpp"
        if self == Backend.CUDA:
            return "cuda"
        if self == Backend.HIP:
            return "hip"
        if self == Backend.OPENCL:
            return "opencl"
        if self == Backend.LAUNCHER:
            return "launcher"


def main(occa_tool: str, data_path: str, backend: Backend, verbose: bool):
    print(f"occa-tool: {occa_tool}, data_path: {data_path}")
    for root, dirs, files in os.walk(data_path):
        # print(f"root: {root}, dirs: {dirs}, files: {files}")
        for file in files:
            if not file.endswith(".cpp") or "ref" in file:
                continue
            file = root + "/" + file
            output_file = file.split(".cpp")[0] + "_ref.cpp"
            exec_str = f"{occa_tool} transpile -b {backend.to_str()} --normalize -i {file} -o {output_file}"
            if verbose:
                print(exec_str)
            os.system(exec_str)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--occa-tool-path", "-o", type=str, required=True)
    parser.add_argument(
        "--data", "-d", type=str, required=True, help="Test data directory path"
    )
    parser.add_argument(
        "--backend", "-b", type=str, required=True, help="serial/openmp/cuda/hip/dpcppp/opencl"
    )
    parser.add_argument(
        "--verbose", "-v", default=False, action="store_const", const=True
    )
    args = parser.parse_args()

    occa_tool = os.path.abspath(args.occa_tool_path)
    data_path = os.path.abspath(args.data)
    backend = Backend.from_str(args.backend)
    main(occa_tool, data_path, backend, args.verbose)
