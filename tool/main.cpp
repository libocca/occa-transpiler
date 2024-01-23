#include <oklt/pipeline/stages/normalizer/normalizer.h>
#include <oklt/core/transpile.h>
#include <oklt/util/io_helper.h>

#include <argparse/argparse.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>

std::string build_output_filename(const std::filesystem::path &input_file_path) {
    std::string out_file = input_file_path.filename().stem().string() + "_transpiled" +
                           input_file_path.filename().extension().string();
    return out_file;
}

const std::map<std::string,  TRANSPILER_TYPE> BACKENDS_MAP = {
    {"cuda", TRANSPILER_TYPE::CUDA},
    {"openmp", TRANSPILER_TYPE::OPENMP}
};

TRANSPILER_TYPE backendFromString(const std::string &type) {
    auto it = BACKENDS_MAP.find(type);
    if(it != BACKENDS_MAP.end()) {
        return it->second;
    }
    throw std::runtime_error("used not registed backend");
}


int main(int argc, char *argv[]) {

    argparse::ArgumentParser program("okl-tool");

    argparse::ArgumentParser normalize_command("normalize");
    normalize_command.add_description("convert OKL 1.0 to OKL 2.0 attributes C++ pure syntax");
    normalize_command.add_argument("-i", "--input")
            .required()
            .help("input file OKL 1.0");
    normalize_command.add_argument("-o", "--output")
            .default_value("")
            .help("optional output file");

    argparse::ArgumentParser transpile_command("transpile");
    transpile_command.add_description("transpile OKL to targeted backend");
    transpile_command.add_argument("-b", "--backend")
            .required()
            //.choices("cuda", "openmp")
            .help("backends: {cuda, openmp}");
    transpile_command.add_argument("-i", "--input")
            .required()
            .help("input file");
    transpile_command.add_argument("--normalize")
            .flag()
            .default_value(false)
            .implicit_value(true)
            .help("should normalize before transpiling");
    transpile_command.add_argument("-o", "--output")
            .default_value("")
            .help("optional output file");

    program.add_subparser(normalize_command);
    program.add_subparser(transpile_command);

    try {
        program.parse_args(argc, argv);
        if(program.is_subcommand_used(normalize_command)) {
            auto input = std::filesystem::path(normalize_command.get("-i"));
            auto output = std::filesystem::path(normalize_command.get("-o"));
            if(output.empty()) {
                output = build_output_filename(input);
            }

            auto input_source = oklt::util::readFileAsStr(input);
            if (!input_source) {
              std::cout << "err: " << input_source.error() << " to read file " << input << '\n';
              return 1;
            }
            oklt::TranspilerSession session{TRANSPILER_TYPE::CUDA};  
            auto normalizedSrc = oklt::normalize({.oklSource = std::move(input_source.value())}, session); 
            if (!normalizedSrc) {
              std::cout << "err to normalize file " << input << '\n';
              return 1;                
            }

            std::cout << "file " << input << " is normalized\n\n" << normalizedSrc.value().cppSource;
            oklt::util::writeFileAsStr(output, normalizedSrc.value().cppSource);

            return 0;
        } else {
            auto source_path = std::filesystem::path(transpile_command.get("-i"));
            auto backend = backendFromString(transpile_command.get("-b"));
            auto need_normalize = transpile_command.get<bool>("--normalize");
            auto output = std::filesystem::path(transpile_command.get("-o"));
            if(output.empty()) {
                output = build_output_filename(source_path);
            }

            std::ifstream ifs(source_path.string());
            std::string sourceCode {std::istreambuf_iterator<char>(ifs), {}};

            oklt::TranspilerInput transpilerParams {
                .sourceCode = sourceCode,
                .sourcePath = source_path,
                .inlcudeDirectories {},
                .defines = {},
                .targetBackend = backend,
                .normalization = need_normalize
            };
            auto ret = oklt::transpile(transpilerParams);
            if(ret) {
              std::cout << "Transpiling success : true" << std::endl;
            } else {
              std::cout << "Transpiling errors: " << std::endl;
              for(const auto &error: ret.error()) {
                std::cout << error.desription << std::endl;
              }
            }
        }
    } catch(const std::exception &ex) {
        std::cout << "Parse arguments: " << ex.what() << std::endl;
        std::cout << program.usage() << std::endl;
    }
    return 0;
}
