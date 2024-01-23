
#include "common/data_directory.h"
#include "load_test_suites.h"
#include "oklt/core/config.h"
#include <map>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;


namespace oklt::tests {


std::vector<std::filesystem::path> loadTestsSuite()
{
    fs::path suitePath = DataRootHolder::instance().suitePath / "suite.json";

    if ( !fs::exists( suitePath ) )
    {
        //Test suites map file was not found
        return { };
    }
    json suite;
    try {
        std::ifstream suiteFile( suitePath );
        suite = json::parse( suiteFile );
    } catch (const std::exception &ex) {
        return {};
    }

    std::vector<std::filesystem::path> resultCases;
    for ( const auto& jsonPath : suite ) {
        resultCases.push_back(DataRootHolder::instance().suitePath / jsonPath.get<std::string>());
    }
    return resultCases;
}   
}
