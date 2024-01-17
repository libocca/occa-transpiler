//TODO: needs implementation
// #include <oklt/normalizer/GnuAttrBasedNormalizer.h>
// #include <oklt/normalizer/MarkerBasedNormalizer.h>

#include "oklt/core/transpile.h"

#include <argparse/argparse.hpp>

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
            //TODO: add implementation here for normalization
            std::cout << "Normalization step is not implemented yet" << std::endl;
            return 0;
        } else {
            auto input = std::filesystem::path(transpile_command.get("-i"));
            auto backend = backendFromString(transpile_command.get("-b"));
            auto need_normalize = transpile_command.get<bool>("--normalize");
            auto output = std::filesystem::path(transpile_command.get("-o"));
            if(output.empty()) {
                output = build_output_filename(input);
            }
            bool ret = okl::transpile(std::cout,
                                      input,
                                      output,
                                      backend,
                                      need_normalize);
            std::cout << "Transpiling success :" << std::boolalpha << ret << std::endl;
        }
    } catch(const std::exception &ex) {
        std::cout << "Parse arguments: " << ex.what() << std::endl;
        std::cout << program.usage() << std::endl;
    }
    return 0;
}
