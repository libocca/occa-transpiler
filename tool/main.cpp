#include <oklt/core/error.h>
#include <oklt/core/transpiler_session/user_input.h>
#include <oklt/core/transpiler_session/user_output.h>

#include <oklt/pipeline/normalizer.h>
#include <oklt/pipeline/normalizer_and_transpiler.h>
#include <oklt/pipeline/transpiler.h>

#include <oklt/util/io_helper.h>

#include <spdlog/spdlog.h>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

std::string build_transpilation_output_filename(const std::filesystem::path& input_file_path) {
    std::string out_file = input_file_path.filename().stem().string() + "_transpiled" +
                           input_file_path.filename().extension().string();
    return out_file;
}

std::string build_launcher_output_filename(const std::filesystem::path& input_file_path) {
    std::string out_file = input_file_path.filename().stem().string() + "_launcher" +
                           input_file_path.filename().extension().string();
    return out_file;
}

std::string build_normalization_output_filename(const std::filesystem::path& input_file_path) {
    std::string out_file = input_file_path.filename().stem().string() + "_normalized" +
                           input_file_path.filename().extension().string();
    return out_file;
}

std::string build_meta_filename(std::filesystem::path file_path) {
    file_path.replace_extension(".json");
    return file_path.string();
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
        .help("backends: {serial, openmp, cuda, hip, dpcpp}");
    transpile_command.add_argument("-i", "--input").required().help("input file");
    transpile_command.add_argument("--normalize")
        .flag()
        .default_value(false)
        .implicit_value(true)
        .help("should normalize before transpiling");
    transpile_command.add_argument("-D", "--define")
        .default_value<std::vector<std::string>>({})
        .append()
        .help("Specify user preprocessor definitions, -D<define>=<content>");
    transpile_command.add_argument("-I", "--include")
        .default_value<std::vector<std::filesystem::path>>({})
        .append()
        .help("Specify user include directories");
    transpile_command.add_argument("-o", "--output")
        .default_value("")
        .help("optional transpilation output file");
    transpile_command.add_argument("-l", "--launcher")
        .default_value("")
        .help("optional launcher output file");
    transpile_command.add_argument("-e", "--external-intrinsic")
        .default_value<std::string>("")
        .help("optional external intrinsic path");
    transpile_command.add_argument("-n", "--normalizer-output")
        .default_value("")
        .help("optional normalization output file");

    program.add_subparser(normalize_command);
    program.add_subparser(transpile_command);

    try {
        program.parse_args(argc, argv);
        if (program.is_subcommand_used(normalize_command)) {
            auto sourcePath = std::filesystem::path(normalize_command.get("-i"));
            auto output = std::filesystem::path(normalize_command.get("-o"));
            if (output.empty()) {
                output = build_normalization_output_filename(sourcePath);
            }

            auto input_source = oklt::util::readFileAsStr(sourcePath);
            if (!input_source) {
                std::cout << "err: " << input_source.error() << " to read file " << sourcePath
                          << '\n';
                return 1;
            }

            auto defines = normalize_command.get<std::vector<std::string>>("-D");
            std::vector<std::string> includesStr;
            if (transpile_command.is_used("-I")) {
                includesStr = transpile_command.get<std::vector<std::string>>("-I");
            }
            std::vector<std::filesystem::path> includes;
            for (const auto& includeStr : includesStr) {
                includes.push_back(includeStr);
            }
            auto result = oklt::normalize({
                .backend = oklt::TargetBackend::CUDA,
                .source = std::move(input_source.value()),
                .sourcePath = sourcePath,
                .includeDirectories = std::move(includes),
                .defines = std::move(defines),
            });
            if (!result) {
                std::cout << "Normalization errors: " << std::endl;
                for (const auto& error : result.error()) {
                    std::cout << error.desc << std::endl;
                }
                std::cout << "err to normalize file " << sourcePath << '\n';
                return 1;
            }

            std::cout << "file " << sourcePath << " is normalized\n\n"
                      << result.value().normalized.source;
            oklt::util::writeFileAsStr(output, result.value().normalized.source);
            // TODO dump headers as well

            return 0;
        } else {
            auto sourcePath = std::filesystem::path(transpile_command.get("-i"));
            auto backend = oklt::backendFromString(transpile_command.get("-b"));
            auto need_normalize = transpile_command.get<bool>("--normalize");

            auto transpilation_output = std::filesystem::path(transpile_command.get("-o"));
            if (transpilation_output.empty()) {
                transpilation_output = build_transpilation_output_filename(sourcePath);
            }
            auto transpilation_meta = build_meta_filename(transpilation_output);

            auto launcher_output = std::filesystem::path(transpile_command.get("-l"));
            if (launcher_output.empty()) {
                launcher_output = build_launcher_output_filename(sourcePath);
            }
            auto launcher_meta = build_meta_filename(launcher_output);

            auto defines = transpile_command.get<std::vector<std::string>>("-D");
            std::vector<std::string> includesStr;
            if (transpile_command.is_used("-I")) {
                includesStr = transpile_command.get<std::vector<std::string>>("-I");
            }
            std::vector<std::filesystem::path> includes;
            for (const auto& includeStr : includesStr) {
                includes.push_back(includeStr);
            }

            auto external_intrinsic = transpile_command.get("-e");
            std::optional<std::filesystem::path> intrinsic_path;
            if (!external_intrinsic.empty()) {
                intrinsic_path = external_intrinsic;
            }

            auto normalization_output = std::filesystem::path(transpile_command.get("-n"));
            if (normalization_output.empty()) {
                normalization_output = build_normalization_output_filename(sourcePath);
            }

            std::ifstream ifs(sourcePath.string());
            std::string sourceCode{std::istreambuf_iterator<char>(ifs), {}};
            oklt::UserInput input{.backend = backend.value(),
                                  .source = sourceCode,
                                  .sourcePath = sourcePath,
                                  .includeDirectories = std::move(includes),
                                  .defines = std::move(defines),
                                  .userIntrincis = intrinsic_path};

            oklt::UserResult result = [](auto&& input, auto need_normalize) {
                if (need_normalize) {
                    return oklt::normalizeAndTranspile(input);
                } else {
                    return oklt::transpile(input);
                }
            }(std::move(input), need_normalize);

            if (result) {
                oklt::UserOutput userOutput = result.value();

                oklt::util::writeFileAsStr(transpilation_output.string(), userOutput.kernel.source);
                oklt::util::writeFileAsStr(transpilation_meta, userOutput.kernel.metadata);

                if (!userOutput.launcher.source.empty()) {
                    oklt::util::writeFileAsStr(launcher_output.string(),
                                               userOutput.launcher.source);
                    oklt::util::writeFileAsStr(launcher_meta, userOutput.launcher.metadata);
                }
                std::cout << result.value().kernel.source;
                SPDLOG_INFO("Transpilation success");
            } else {
                SPDLOG_ERROR("Transpilation failed");
                for (const auto& error : result.error()) {
                    std::cerr << error.desc << "\n";
                }
            }
        }
    } catch (const std::exception& ex) {
        std::cout << "Parse arguments: " << ex.what() << std::endl;
        std::cout << program.usage() << std::endl;
    }
    return 0;
}
