# How to add new backend in occa-transpiler 
In this short tutorial, you will learn how to add your own new backend. The example will be shown in OpenCL.

## Prepartion
Before you start adding a new backend to the project, you need to add a new backend to the target_backends.cpp file at the address lib/core/target_backends.cpp. The following changes need to be made:
```cpp

tl::expected<TargetBackend, std::string> backendFromString(const std::string& type) {
    static const std::map<std::string, TargetBackend> BACKENDS_MAP = {
        ...
        {"opencl", TargetBackend::OPENCL},
        ...
    };

    auto it = BACKENDS_MAP.find(util::toLower(type));
    if (it != BACKENDS_MAP.end()) {
        return it->second;
    }
    return tl::unexpected("unknown backend is requested");
}

std::string backendToString(TargetBackend backend) {
    switch (backend) {
        ...
        case TargetBackend::OPENCL:
            return std::string{"opencl"};
        ...
    }
    return {};
}

```
And according to the backend we add, we need to add it to either isHostCategory or isDeviceCategory. Since OpenCl refers to the second case, we make the following changes:
```cpp
bool isDeviceCategory(TargetBackend backend) {
    switch (backend) {
        ...
        case TargetBackend::OPENCL:
            return true;
        default:
            return false;
    }
}
```

## Implementaion of backend 
### Adding new files
At the following address: lib/attributes/backend/, create a folder with the name of the backend and add the following files:
- atomic.cpp
- barrier.cpp
- exclusive.cpp
- global_constant.cpp
- global_function.cpp
- inner.cpp
- kernel.cpp
- outer.cpp
- restrict.cpp
- shared.cpp
- tile.cpp
- translation_unit.cpp

And additional ones if needed:
- common.h
- common.cpp

### Connection of new files
After creating the folder and creating the appropriate ones, add the stem files to CMakeLists.txt located at the following address: lib/CMakeLists.txt 
```txt
    # OPENCL
    attributes/backend/opencl/kernel.cpp
    attributes/backend/opencl/translation_unit.cpp
    attributes/backend/opencl/global_constant.cpp
    attributes/backend/opencl/global_function.cpp
    attributes/backend/opencl/outer.cpp
    attributes/backend/opencl/inner.cpp
    attributes/backend/opencl/tile.cpp
    attributes/backend/opencl/shared.cpp
    attributes/backend/opencl/restrict.cpp
    attributes/backend/opencl/atomic.cpp
    attributes/backend/opencl/barrier.cpp
    attributes/backend/opencl/exclusive.cpp
    attributes/backend/opencl/common.cpp
    attributes/backend/opencl/common.h
```

### Stucture of files
Other backends can be used as templates for writing files. For example, here is a ready-made implementation of @atomic for opencl in atomic.cpp:
```cpp
namespace {
using namespace oklt;
using namespace clang;

HandleResult handleAtomicAttribute(SessionStage& stage, const Stmt& stmt, const Attr& attr) {
    SPDLOG_DEBUG("Handle [@atomic] attribute (stmt)");

    removeAttribute(stage, attr);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = registerBackendHandler(
        TargetBackend::OPENCL, ATOMIC_ATTR_NAME, handleAtomicAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENCL] Failed to register {} attribute handler", ATOMIC_ATTR_NAME);
    }
}
} 
```

The main thing is to change the TargetBackend and SPDLOG_ERROR to the appropriate one for the backend when writing implementations or using other files as templates.


## Test run of the program 
To check the implementation of the backend, you need to run it on an example. To do this, we create a file in which we will add the test code for testing, so we created test.cpp in the example folder.  

To run the program and check the code from the file, run the following command:

``` bash
./build/bin/occa-tool transpile --normalize -b opencl -i $FullPath$/occa-transpiler/example/test.cpp -o $FullPath$/occa-transpiler/example/test-out.cpp --sema with-sema 
```

## Tests for backend 
This chapter will show you the necessary steps to create the appropriate tests, generate them, and run them.

### Adding of new tests
Creating tests to check the backend consists of two stages. The first step is to create the tests themselves, and the second is to create the corresponding json configuration files.

Let's consider the first stage. To add tests, go to tests/functional/data/transpiler/backends/ and create an opencl folder. In this folder, add folders for tests, and add example files to the folders themselves. Let's consider an example of creating a test for @barrier, to do this, create the nobarrier folder and create the nobarrier_builtin file.cpp 
```cpp
@kernel void hello_kern() {
    for (int i = 0; i < 10; ++i; @outer) {
        @shared int shm[10];
        for (int j = 0; j < 10; ++j; @inner) {
            shm[j] = j;
        }

        @nobarrier for (int j = 0; j < 10; ++j; @inner) {
            shm[j] = j;
        }

        for (int j = 0; j < 10; ++j; @inner) {
            shm[j] = j;
        }

        for (int j = 0; j < 10; ++j; @inner) {
            shm[j] = j;
        }
    }
}

@kernel void priority_issue() {
    @outer for (int i = 0; i < 32; ++i) {
        @shared float shm[32];
        @nobarrier for (int j = 0; j < 32; ++j; @inner) {
                shm[i] = i;
        }
        @inner for (int j = 0; j < 32; ++j) {
                @atomic shm[i * j] += 32;
        }
    }
}
```

Let's move on to the second step. Now you need to go to tests/functional/configs/test_suite_transpiler/backends/ and create the opencl folder. Create the file nobarrier.json

```json
[
  {
    "action": "normalize_and_transpile",
    "action_config": {
      "backend": "opencl",
      "source": "transpiler/backends/opencl/nobarrier/nobarrier_builtin.cpp",
      "includes": [],
      "defs": [],
      "launcher": ""
    },
    "reference": "transpiler/backends/opencl/nobarrier/nobarrier_builtin_ref.cpp"
  }
]

```

### Adding python script for test regeneration
ДFor the created tests, you need to create the appropriate reference files. To do this, you can use the python script located at the following address: script/regenerate_tests_ref.py, but it needs to be slightly modified by adding the newly created backend. The following changes are made:
```python
    SERIAL = 0
    OPENMP = 1
    CUDA = 2
    HIP = 3
    DPCPP = 4
    OPENCL = 5
    LAUNCHER = 6
```
Add new options to the following functions:
```python
 def from_str(s: str) -> "Backend":
    s = s.lower()
    ...
    if s == "opencl":
        return Backend.OPENCL
    ...

 def to_str(self) -> str:
    ...
    if self == Backend.OPENCL:
            return "opencl"
    ...
```
And change the selection options:
```python
    parser.add_argument(
        "--backend", "-b", type=str, required=True, help="serial/openmp/cuda/hip/dpcppp/opencl"
    )
```

### Creation of new test
After creating the tests and modifying the script, let's generate the reference files using the following command:
``` bash
python3 ./script/regenerate_test_ref.py -o ./build/bin/occa-tool -d test/functional/data/transpiler/backends/opencl/ -b opencl 
```

### Test run
To run all tests, use the following command:
``` bash
./occa-transpiler-tests --suite configs/test_suite_transpiler/backends/opencl/ --data_root data/
```
## Tips

### Hooks
Hooks are a mechanism that allows you to register functions (usually known as handlers or event handlers) that will be called when certain AST nodes are encountered during code analysis. This allows you to intervene in the AST analysis and processing process to perform various tasks, in our case code generation.

The general structure of the hook is as follows:
```cpp
__attribute__((constructor)) void functionName() {
    auto ok =
        registerBackendHandler(TargetBackend::<beckend>, <hook target>, <functionToCall>);

    if (!ok) {
        SPDLOG_ERROR("[<beckend>] Failed to register {} attribute handler", <hook target>);
    }
}
```
The program has the following options for hook targets:
- KERNEL_ATTR_NAME
- OUTER_ATTR_NAME
- TILE_ATTR_NAME
- SHARED_ATTR_NAME
- RESTRICT_ATTR_NAME
- BARRIER_ATTR_NAME
- NO_BARRIER_ATTR_NAME
- EXCLUSIVE_ATTR_NAME
- ATOMIC_ATTR_NAME

Let's look at an example of using a hook on the example of implementing @barrier for OpenCL: 

```cpp
__attribute__((constructor)) void registerAttrBackend() {
    auto ok =
        registerBackendHandler(TargetBackend::OPENCL, BARRIER_ATTR_NAME, handleBarrierAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENCL] Failed to register {} attribute handler", BARRIER_ATTR_NAME);
    }
}
```
As you can see, the following two positions responsible for global function and global constant are missing from the provided hook targets. In this case, the hook structure is as follows: 

```cpp
__attribute__((constructor)) void functionName() {
    auto ok =
        registerBackendHandler(TargetBackend::<beckend>, <functionToCall>);

    if (!ok) {
        SPDLOG_ERROR("[<beckend>] Failed to register {} attribute handler");
    }
}
```
This case can be seen in the example of the global const implementation for OpenCl:

```cpp
__attribute__((constructor)) void registeCUDAGlobalConstantHandler() {
    auto ok = registerImplicitHandler(TargetBackend::OPENCL, handleGlobalConstant);

    if (!ok) {
        SPDLOG_ERROR("[OPENCL] Failed to register implicit handler for global constant");
    }
}
```

### Rewriter
This is one of the main tools used in the backend implementation. The general structure can be found here []. 

When using it, keep in mind that inserting it when performing a replace or delete operation can lead to errors. Therefore, you need to be sure of the cursor position when using insert or replace operations.

## Tips for debug
To improve the code debugging process, you can “disable” the format function located at the following address: lib/core/utils/format.cpp. This function processes the text before outputting it into a more human-readable format, but in turn complicates the debugging process. To do this, use the format:
```cpp
std::string format(std::string_view code) {
    const std::vector<Range> ranges(1, Range(0, code.size()));
    auto style = format::getLLVMStyle();
    style.MaxEmptyLinesToKeep = 1;
    style.SeparateDefinitionBlocks = format::FormatStyle::SeparateDefinitionStyle::SDS_Always;

    Replacements replaces = format::reformat(style, code, ranges);
    auto changedCode = applyAllReplacements(code, replaces);
    if (!changedCode) {
        SPDLOG_ERROR("{}", toString(changedCode.takeError()));
        return {};
    }
    return changedCode.get();
}
```
function must be edited to this format:
```cpp
std::string format(std::string_view code) {
    return std::string(code);
}
```

