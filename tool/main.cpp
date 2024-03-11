#include <oklt/core/error.h>
#include <oklt/core/transpiler_session/user_input.h>
#include <oklt/core/transpiler_session/user_output.h>

#include <oklt/pipeline/normalizer.h>
#include <oklt/pipeline/normalizer_and_transpiler.h>
#include <oklt/pipeline/transpiler.h>

#include <oklt/util/io_helper.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

std::string build_transpilation_output_filename(const std::filesystem::path& input_file_path) {
    std::string out_file = input_file_path.filename().stem().string() + "_transpiled" +
                           input_file_path.filename().extension().string();
    return out_file;
}

std::string build_normalization_output_filename(const std::filesystem::path& input_file_path) {
    std::string out_file = input_file_path.filename().stem().string() + "_normalized" +
                           input_file_path.filename().extension().string();
    return out_file;
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("okl-tool");

    argparse::ArgumentParser normalize_command("normalize");
    normalize_command.add_description("convert OKL 1.0 to OKL 2.0 attributes C++ pure syntax");
    normalize_command.add_argument("-i", "--input").required().help("input file OKL 1.0");
    normalize_command.add_argument("-o", "--output").default_value("").help("optional output file");
    normalize_command.add_argument("-D", "--define")
        .default_value<std::vector<std::string>>({})
        .append()
        .help("Specify user preprocessor definitions");
    normalize_command.add_argument("-I", "--include")
        .default_value<std::vector<std::string>>({})
        .append()
        .help("Specify user include directories");

    argparse::ArgumentParser transpile_command("transpile");
    transpile_command.add_description("transpile OKL to targeted backend");
    transpile_command.add_argument("-b", "--backend")
        .required()
        //.choices("cuda", "openmp")
        .help("backends: {cuda, hip, dpcpp, openmp}");
    transpile_command.add_argument("-i", "--input").required().help("input file");
    transpile_command.add_argument("--normalize")
        .flag()
        .default_value(false)
        .implicit_value(true)
        .help("should normalize before transpiling");
    transpile_command.add_argument("-D", "--define")
        .default_value<std::vector<std::string>>({})
        .append()
        .help("Specify user preprocessor definitions");
    transpile_command.add_argument("-I", "--include")
        .default_value<std::vector<std::filesystem::path>>({})
        .append()
        .help("Specify user include directories");
    transpile_command.add_argument("-o", "--output")
        .default_value("")
        .help("optional transpilation output file");
    transpile_command.add_argument("-n", "--normalizer-output")
        .default_value("")
        .help("optional normalization output file");
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
            auto output = std::filesystem::path(normalize_command.get("-n"));
            if (output.empty()) {
                output = build_normalization_output_filename(input);
            }

            auto input_source = oklt::util::readFileAsStr(input);
            if (!input_source) {
                std::cout << "err: " << input_source.error() << " to read file " << input << '\n';
                return 1;
            }

            auto defines = normalize_command.get<std::vector<std::string>>("-D");
            auto includes = normalize_command.get<std::vector<std::filesystem::path>>("-I");

            auto result = oklt::normalize({
                .backend = oklt::TargetBackend::CUDA,
                .sourceCode = std::move(input_source.value()),
                .inlcudeDirectories = std::move(includes),
                .defines = std::move(defines),
            });
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
            auto transpilation_output = std::filesystem::path(transpile_command.get("-o"));
            if (transpilation_output.empty()) {
                transpilation_output = build_transpilation_output_filename(source_path);
            }

            auto defines = transpile_command.get<std::vector<std::string>>("-D");
            std::vector<std::string> includesStr;
            if (transpile_command.is_used("-I")) {
                includesStr = transpile_command.get<std::vector<std::string>>("-I");
            }
            std::vector<std::filesystem::path> includes;
            for (const auto& includeStr : includesStr) {
                includes.push_back(includeStr);
            }

            auto normalization_output = std::filesystem::path(transpile_command.get("-n"));
            if (normalization_output.empty()) {
                normalization_output = build_normalization_output_filename(source_path);
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
                                  .inlcudeDirectories = std::move(includes),
                                  .defines = std::move(defines)};

            oklt::UserResult result = [](auto&& input, auto need_normalize) {
                if (need_normalize) {
                    return oklt::normalizeAndTranspile(input);
                } else {
                    return oklt::transpile(input);
                }
            }(std::move(input), need_normalize);

            if (result) {
                oklt::UserOutput userOutput = result.value();
                // oklt::util::writeFileAsStr(normalization_output.string(),
                // userOutput.normalized.sourceCode);
                oklt::util::writeFileAsStr(transpilation_output.string(),
                                           userOutput.kernel.sourceCode);
                std::cout << "Transpiling success : true" << std::endl;
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
