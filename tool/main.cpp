#include <oklt/core/error.h>
#include <oklt/core/transpiler_session/transpiler_session.h>

#include <oklt/pipeline/normalizer.h>
#include <oklt/pipeline/normalizer_and_transpiler.h>
#include <oklt/pipeline/transpiler.h>

#include <oklt/util/io_helper.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

std::string build_output_filename(const std::filesystem::path& input_file_path) {
    std::string out_file = input_file_path.filename().stem().string() + "_transpiled" +
                           input_file_path.filename().extension().string();
    return out_file;
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("okl-tool");

    argparse::ArgumentParser normalize_command("normalize");
    normalize_command.add_description("convert OKL 1.0 to OKL 2.0 attributes C++ pure syntax");
    normalize_command.add_argument("-i", "--input").required().help("input file OKL 1.0");
    normalize_command.add_argument("-o", "--output").default_value("").help("optional output file");

    argparse::ArgumentParser transpile_command("transpile");
    transpile_command.add_description("transpile OKL to targeted backend");
    transpile_command.add_argument("-b", "--backend")
        .required()
        //.choices("cuda", "openmp")
        .help("backends: {cuda, openmp}");
    transpile_command.add_argument("-i", "--input").required().help("input file");
    transpile_command.add_argument("--normalize")
        .flag()
        .default_value(false)
        .implicit_value(true)
        .help("should normalize before transpiling");
    transpile_command.add_argument("-o", "--output").default_value("").help("optional output file");
    transpile_command.add_argument("-s", "--sema")
        .help("sema: {no-sema, with-sema}")
        .required()
        .default_value("with-sema");

    program.add_subparser(normalize_command);
    program.add_subparser(transpile_command);

    try {
        program.parse_args(argc, argv);
        if (program.is_subcommand_used(normalize_command)) {
            auto input = std::filesystem::path(normalize_command.get("-i"));
            auto output = std::filesystem::path(normalize_command.get("-o"));
            if (output.empty()) {
                output = build_output_filename(input);
            }

            auto input_source = oklt::util::readFileAsStr(input);
            if (!input_source) {
                std::cout << "err: " << input_source.error() << " to read file " << input << '\n';
                return 1;
            }

            auto result = oklt::normalize({.backend = oklt::TargetBackend::CUDA,
                                           .sourceCode = std::move(input_source.value())});
            if (!result) {
                std::cout << "Normalization errors: " << std::endl;
                for (const auto& error : result.error()) {
                    std::cout << error.desc << std::endl;
                }
                std::cout << "err to normalize file " << input << '\n';
                return 1;
            }

            std::cout << "file " << input << " is normalized\n\n"
                      << result.value().normalized.sourceCode;
            oklt::util::writeFileAsStr(output, result.value().normalized.sourceCode);

            return 0;
        } else {
            auto source_path = std::filesystem::path(transpile_command.get("-i"));
            auto backend = oklt::backendFromString(transpile_command.get("-b"));
            auto need_normalize = transpile_command.get<bool>("--normalize");
            auto output = std::filesystem::path(transpile_command.get("-o"));
            if (output.empty()) {
                output = build_output_filename(source_path);
            }

            oklt::AstProcessorType procType = [&]() {
                auto semaType = transpile_command.get("-s");
                if (semaType == "with-sema") {
                    return oklt::AstProcessorType::OKL_WITH_SEMA;
                }
                return oklt::AstProcessorType::OKL_NO_SEMA;
            }();

            std::ifstream ifs(source_path.string());
            std::string sourceCode{std::istreambuf_iterator<char>(ifs), {}};
            oklt::UserInput input{.backend = backend.value(),
                                  .astProcType = procType,
                                  .sourceCode = sourceCode,
                                  .sourcePath = source_path,
                                  .inlcudeDirectories = {},
                                  .defines = {}};

            oklt::UserResult result = [](auto&& input, auto need_normalize) {
                if (need_normalize) {
                    return oklt::normalizeAndTranspile(input);
                } else {
                    return oklt::transpile(input);
                }
            }(std::move(input), need_normalize);

            if (result) {
                auto &val = result.value();
                std::cout << "Transpiling success : true" << std::endl;
                std::cout << "Transpiled Source:" << std::endl;
                std::cout << val.kernel.sourceCode;
                if (!val.launcher.sourceCode.empty()) {
                    std::cout << "Launcher Source:" << std::endl;
                    std::cout << val.launcher.sourceCode;
                }
            } else {
                std::cout << "Transpiling errors: " << std::endl;
                for (const auto& error : result.error()) {
                    std::cout << error.desc << std::endl;
                }
            }
        }
    } catch (const std::exception& ex) {
        std::cout << "Parse arguments: " << ex.what() << std::endl;
        std::cout << program.usage() << std::endl;
    }
    return 0;
}
