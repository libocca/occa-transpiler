# OCCA Transpiler

## Requirements
**Minimum:**
- CMake v3.26 or newer
- C++17 compiler
- C11 compiler


## Build
### Setup llvm 
```bash
wget https://raw.githubusercontent.com/opencollab/llvm-jenkins.debian.net/master/llvm.sh
sudo ./llvm.sh 17 all
rm llvm.sh
```

### Compile
```bash
mkdir -p build && cd build
cmake .. && make -j$(($(nproc)-2))
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