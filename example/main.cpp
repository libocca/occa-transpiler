#include <filesystem>
#include <fstream>
#include <sstream>
#include "oklt/core/error.h"
#include "oklt/core/target_backends.h"
#include "oklt/pipeline/normalizer_and_transpiler.h"

#include <iostream>

using namespace oklt;

int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string filePath(argv[1]);
        std::ifstream ifs(filePath);
        if (ifs) {
            std::string sourceCode{std::istreambuf_iterator<char>(ifs), {}};
            UserInput input{.backend = TargetBackend::CUDA,
                            .astProcType = AstProcessorType::OKL_WITH_SEMA,
                            .sourceCode = std::move(sourceCode),
                            .sourcePath = std::filesystem::path(filePath),
                            .inlcudeDirectories = {},
                            .defines = {}};
            auto result = normalizeAndTranspile(std::move(input));
            if (!result) {
                std::stringstream ss;
                for (const auto& err : result.error()) {
                    ss << err.desc << std::endl;
                }
                std::cout << "Transpilation to CUDA backend errors occur :" << ss.str()
                          << std::endl;
                return 1;
            }
            std::cout << result.value().kernel.sourceCode << std::endl;
        }
    }

    return 0;
}
