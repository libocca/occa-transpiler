
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


std::vector<GenericSuite> loadTestsSuite()
{
    fs::path suiteMapPath = DataRootHolder::instance().suitePath / "test_suite_map.json";

    if ( !fs::exists( suiteMapPath ) )
    {
        //Test suites map file was not found
        return { };
    }
    json suiteMap;
    try {
        std::ifstream suiteMapFile( suiteMapPath );
        suiteMap = json::parse( suiteMapFile );
    } catch (const std::exception &ex) {
        return {};
    }

    std::vector<GenericSuite> suites;
    for ( const auto& entry : suiteMap )
    {
        auto backend = entry[0].get<std::string>();
        auto expectedBackend = backendFromString(backend);
        if(expectedBackend) {
            GenericSuite suite {expectedBackend.value(), std::list<std::filesystem::path> {}};
            for(const auto &filePath: entry[0].get<std::vector<std::string>>()) {
                std::filesystem::path p(filePath);
                suite.testCaseFiles.push_back(p);
            }
        }
    }
    return suites;
}   
}
