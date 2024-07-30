# OCCA Transpiler

## Requirements
**Minimum:**
- CMake v3.26 or newer
- C++17 compiler
- C11 compiler
- Clang 17(exactly)



## Build
### Setup clang 17
The current version of OCCA transpiler requires exactly clang 17. In future the transpiler will be updated to have compatibility layer to support newer versions of clang.
#### Install clang 17
```bash
wget https://raw.githubusercontent.com/opencollab/llvm-jenkins.debian.net/master/llvm.sh
sudo ./llvm.sh 17 all
rm llvm.sh
```

#### Build clang from source
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-17.0.6
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS='clang' -DCMAKE_INSTALL_PREFIX=<clang_install_prefix> -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_PARALLEL_LINK_JOBS=1
ninja -C build -j$(nproc)
ninja -C build install
```
**Note:** Replace `<clang_install_prefix>` with the desired installation path. \
**Note:** Replace `ninja` with `make` if you are using `make` as the build system. \
**Note:** In case of 'make' build system, add `-j$(nproc)` with `-jN` where `N` is the number of cores you want to use for the build \
          If your system has less than 32GB of RAM, it is recommended to use `-j1` instead of `-j$(nproc)` to avoid running out of memory on linkage step.

### Submodules
```bash
git submodule init
git submodule update
```

### Compile
```bash
mkdir -p build && cd build
cmake .. && make -j$(($(nproc)-2))
```

If you use clang 17 built from source please specify path to it via option -DOCCA_LOCAL_CLANG_PATH=<clang_install_prefix>:
```bash
mkdir -p build && cd build
cmake -DOCCA_LOCAL_CLANG_PATH=<clang_install_prefix> .. && make -j$(($(nproc)-2))
```

## OCCA-Tool
### Normalize
```bash
Usage: normalize [--help] [--version] --input VAR [--output VAR]

convert OKL 1.0 to OKL 2.0 attributes C++ pure syntax

Optional arguments:
  -h, --help     shows help message and exits 
  -v, --version  prints version information and exits 
  -i, --input    input file OKL 1.0 [required]
  -o, --output   optional output file [nargs=0..1] [default: ""]
```

### Transpile 
```bash
Usage: transpile [--help] [--version] --backend VAR --input VAR [--normalize] [--output VAR]

transpile OKL to targeted backend

Optional arguments:
  -h, --help     shows help message and exits 
  -v, --version  prints version information and exits 
  -b, --backend  backends: {cuda, openmp} [required]
  -i, --input    input file [required]
  --normalize    should normalize before transpiling 
  -o, --output   optional output file [nargs=0..1] [default: ""]
```

### Logging
Logging level can be set with `OKLT_LOG_LEVEL` enviroment variable.

Possible values:
- trace
- debug
- info
- warn
- err
- critical


## [Documentation](./docs/README.md)
