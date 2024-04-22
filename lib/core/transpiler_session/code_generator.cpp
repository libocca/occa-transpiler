#include <oklt/util/io_helper.h>

#include "core/transpiler_session/code_generator.h"
#include "core/transpiler_session/header_info.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpilation_node.h"
#include "core/transpiler_session/transpiler_session.h"

#include "core/attribute_manager/attribute_manager.h"

#include "core/utils/attributes.h"
#include "core/vfs/overlay_fs.h"

#include <clang/AST/Attr.h>
#include <clang/FrontendTool/Utils.h>
#include <clang/Lex/PreprocessorOptions.h>

#include <spdlog/spdlog.h>
namespace {
using namespace oklt;
using namespace clang;

HandleResult applyTranspilationToAttrNode(SessionStage& stage,
                                          const DynTypedNode& node,
                                          const Attr& attr) {
    auto& am = stage.getAttrManager();
    auto params = am.parseAttr(stage, attr);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    if (ASTNodeKind::getFromNodeKind<Decl>().isBaseOf(node.getNodeKind())) {
        return am.handleAttr(stage, *node.get<Decl>(), attr, &params.value());
    }

    if (ASTNodeKind::getFromNodeKind<Stmt>().isBaseOf(node.getNodeKind())) {
        return am.handleAttr(stage, *node.get<Stmt>(), attr, &params.value());
    }

    return tl::make_unexpected(
        Error{{}, std::string("unexpected node kind:") + node.getNodeKind().asStringRef().str()});
}

HandleResult applyTranspilationToNode(SessionStage& stage, const DynTypedNode& node) {
    if (ASTNodeKind::getFromNodeKind<Decl>().isBaseOf(node.getNodeKind())) {
        return AttributeManager::instance().handleNode(stage, *node.get<Decl>());
    }

    if (ASTNodeKind::getFromNodeKind<Stmt>().isBaseOf(node.getNodeKind())) {
        return AttributeManager::instance().handleNode(stage, *node.get<Stmt>());
    }

    return tl::make_unexpected(
        Error{{}, std::string("unexpected node kind:") + node.getNodeKind().asStringRef().str()});
}

HandleResult applyTranspilationToNode(SessionStage& stage,
                                      const DynTypedNode& node,
                                      const Attr* attr) {
    SPDLOG_TRACE("{} node name; {}", __PRETTY_FUNCTION__, node.getNodeKind().asStringRef());
    if (!attr) {
        return applyTranspilationToNode(stage, node);
    }

    return applyTranspilationToAttrNode(stage, node, *attr);
}

HandleResult applyTranspilationToNodes(SessionStage& stage, const TranspilationNodes& nodes) {
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    for (const auto& tnode : nodes) {
        // set appropriate parsed KernelInfo and LoopInfo as active for current node
        sema.setParsedKernelInfo(tnode.ki);
        sema.setLoopInfo(tnode.li);
        auto result = applyTranspilationToNode(stage, tnode.node, tnode.attr);
        if (!result) {
            result.error().ctx = tnode.attr ? tnode.attr->getRange() : SourceRange();
            return result;
        }
    }

    return {};
}

// remove system header to avoid insertion of them during fusion of final transpiled kernel
void removeSystemHeaders(SessionStage& stage, const HeaderDepsInfo& deps) {
    auto& rewriter = stage.getRewriter();
    for (const auto& dep : deps.topLevelDeps) {
        if (!SrcMgr::isSystem(dep.fileType)) {
            continue;
        }
        SPDLOG_TRACE("remove system include {} {}", dep.relativePath, dep.fileName);
        rewriter.RemoveText({dep.hashLoc, dep.filenameRange.getEnd()});
    }
}

// gather all transpiled files: main input and affected header and also header with removed system
// includes
TransformedFiles gatherTransformedFiles(SessionStage& stage) {
    auto inputs = stage.getRewriterResultForHeaders();
    // merging operation move the source to destination map so clone headers
    // to preserve them for possible laucher generator
    auto clone = stage.getSession().getInput().headers;
    inputs.fileMap.merge(clone);
    inputs.fileMap["okl_kernel.cpp"] = stage.getRewriterResultForMainFile();
    return inputs;
}

tl::expected<std::string, Error> preprocessedInputs(SessionStage& stage,
                                                    const TransformedFiles& inputs) {
    auto invocation = std::make_shared<CompilerInvocation>();

    auto& ppOutOpt = invocation->getPreprocessorOutputOpts();
    ppOutOpt.ShowCPP = true;
    ppOutOpt.ShowLineMarkers = false;
    ppOutOpt.ShowIncludeDirectives = false;

    const std::string FUSED_KERNEL_FILENAME_BASE = "fused_inc_kernel";
    std::time_t ct = std::time(0);
    std::string outputFileName = FUSED_KERNEL_FILENAME_BASE + ctime(&ct) + ".cpp";
    invocation->getFrontendOpts().OutputFile = outputFileName;

    // set options from parent compiler
    invocation->getHeaderSearchOpts() = stage.getCompiler().getHeaderSearchOpts();
    invocation->getPreprocessorOpts() = stage.getCompiler().getPreprocessorOpts();
    // TODO get this info from user input aka json prop file
    invocation->getDiagnosticOpts().Warnings = {"no-extra-tokens", "no-invalid-pp-token"};

    invocation->getFrontendOpts().Inputs.push_back(
        FrontendInputFile("okl_kernel.cpp", Language::CXX));
    invocation->getFrontendOpts().ProgramAction = frontend::PrintPreprocessedInput;
    invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";

    CompilerInstance compiler;
    compiler.setInvocation(std::move(invocation));
    compiler.createDiagnostics();
    compiler.createFileManager(makeOverlayFs(
        stage.getCompiler().getFileManager().getVirtualFileSystemPtr(), inputs.fileMap));

    // XXX clang PrintPreprocessedInput action currently can provide output in two ways:
    //     - print it into STDOUT
    //     - write to the file
    //     second addional option  is used to dump output into FS and then
    //     read/delete it
    //
    if (!ExecuteCompilerInvocation(&compiler)) {
        std::filesystem::remove(outputFileName);
        return tl::make_unexpected(Error{{}, "failed to make preprocessing okl_kernel.cpp: "});
    }

    auto preprocessedAndFused = util::readFileAsStr(outputFileName);
    if (!preprocessedAndFused) {
        return tl::make_unexpected(Error{{}, "failed to read file " + outputFileName});
    }
    std::filesystem::remove(outputFileName);

    return preprocessedAndFused.value();
}

std::string restoreSystemAndBackendHeaders(std::string& input, const HeaderDepsInfo& deps) {
    // insert backend specific headers and namespaces
    for (auto it = deps.backendNss.rbegin(); it < deps.backendNss.rend(); ++it) {
        input.insert(0, *it);
    }

    for (auto it = deps.backendHeaders.rbegin(); it < deps.backendHeaders.rend(); ++it) {
        input.insert(0, *it);
    }

    // restore system headers
    for (const auto& dep : deps.topLevelDeps) {
        if (!clang::SrcMgr::isSystem(dep.fileType)) {
            continue;
        }
        input.insert(0, "#include <" + dep.fileName + ">\n");
    }

    return input;
}

tl::expected<std::string, Error> fuseIncludeDeps(SessionStage& stage, const HeaderDepsInfo& deps) {
    removeSystemHeaders(stage, deps);

    auto inputs = gatherTransformedFiles(stage);

    auto preprocessedResult = preprocessedInputs(stage, inputs);
    if (!preprocessedResult) {
        return preprocessedResult;
    }

    auto finalTranspiledKernel = restoreSystemAndBackendHeaders(preprocessedResult.value(), deps);
    return finalTranspiledKernel;
}
}  // namespace

namespace oklt {
tl::expected<std::string, Error> generateTranspiledCode(SessionStage& stage) {
    const auto& nodes = stage.tryEmplaceUserCtx<TranspilationNodes>();
    auto result = applyTranspilationToNodes(stage, nodes);
    if (!result) {
        return tl::make_unexpected(std::move(result.error()));
    }

    const auto& deps = stage.tryEmplaceUserCtx<HeaderDepsInfo>();
    auto finalResult = fuseIncludeDeps(stage, deps);
    if (!finalResult) {
        return finalResult;
    }

    return std::move(finalResult.value());
}

tl::expected<std::string, Error> generateTranspiledCodeMetaData(SessionStage& stage) {
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto programMeta = sema.getProgramMetaData();
    nlohmann::json kernel_metadata;
    to_json(kernel_metadata, programMeta);
    auto kernelMetaData = kernel_metadata.dump(2);

    SPDLOG_DEBUG("Program metadata: {}", kernelMetaData);

    return kernelMetaData;
}
}  // namespace oklt
