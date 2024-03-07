from argparse import ArgumentParser
import os
from enum import Enum


class Backend(Enum):
    DPCPP = 0
    OPENMP = 1
    CUDA = 2
    HIP = 3

    def from_str(s: str) -> "Backend":
        s = s.lower()
        if s == "dpcpp":
            return Backend.DPCPP
        if s == "openmp":
            return Backend.OPENMP
        if s == "cuda":
            return Backend.CUDA
        if s == "hip":
            return Backend.HIP

    def to_str(self) -> str:
        if self == Backend.DPCPP:
            return "dpcpp"
        if self == Backend.OPENMP:
            return "openmp"
        if self == Backend.CUDA:
            return "cuda"
        if self == Backend.HIP:
            return "hip"


def main(occa_tool: str, data_path: str, backend: Backend):
    print(f"occa-tool: {occa_tool}, data_path: {data_path}")
    for root, dirs, files in os.walk(data_path):
        # print(f"root: {root}, dirs: {dirs}, files: {files}")
        for file in files:
            if not file.endswith(".cpp") or "ref" in file:
                continue
            file = root + "/" + file
            output_file = file.split(".cpp")[0] + "_ref.cpp"
            os.system(
                f"{occa_tool} transpile -b {backend.to_str()} --normalize -i {file} -o {output_file}"
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--occa-tool-path", "-o", type=str, required=True)
    parser.add_argument(
        "--data", "-d", type=str, required=True, help="Test data directory path"
    )
    parser.add_argument(
        "--backend", "-b", type=str, required=True, help="dpcppp/openmp/cuda/hip"
    )
    args = parser.parse_args()

    occa_tool = os.path.abspath(args.occa_tool_path)
    data_path = os.path.abspath(args.data)
    backend = Backend.from_str(args.backend)
    main(occa_tool, data_path, backend)
