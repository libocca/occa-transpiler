#include <oklt/util/io_helper.h>

#include "core/transpiler_session/code_generator.h"
#include "core/transpiler_session/header_info.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpilation_node.h"
#include "core/transpiler_session/transpiler_session.h"

#include "core/attribute_manager/attribute_manager.h"

#include "core/vfs/overlay_fs.h"

#include <clang/AST/Attr.h>
#include <clang/FrontendTool/Utils.h>
#include <clang/Lex/PreprocessorOptions.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult applyTranspilationToAttrNode(const Attr& attr,
                                          const DynTypedNode& node,
                                          SessionStage& stage) {
    auto& am = stage.getAttrManager();
    auto params = am.parseAttr(attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    if (ASTNodeKind::getFromNodeKind<Decl>().isBaseOf(node.getNodeKind())) {
        return am.handleAttr(attr, *node.get<Decl>(), &params.value(), stage);
    }

    if (ASTNodeKind::getFromNodeKind<Stmt>().isBaseOf(node.getNodeKind())) {
        return am.handleAttr(attr, *node.get<Stmt>(), &params.value(), stage);
    }

    return tl::make_unexpected(
        Error{{}, std::string("unexpected node kind:") + node.getNodeKind().asStringRef().data()});
}

HandleResult applyTranspilationToNode(const DynTypedNode& node, SessionStage& stage) {
    if (ASTNodeKind::getFromNodeKind<Decl>().isBaseOf(node.getNodeKind())) {
        return AttributeManager::instance().handleNode(*node.get<Decl>(), stage);
    }

    if (ASTNodeKind::getFromNodeKind<Stmt>().isBaseOf(node.getNodeKind())) {
        return AttributeManager::instance().handleNode(*node.get<Stmt>(), stage);
    }

    return tl::make_unexpected(
        Error{{}, std::string("unexpected node kind:") + node.getNodeKind().asStringRef().data()});
}

HandleResult applyTranspilationToNode(const Attr* attr,
                                      const DynTypedNode& node,
                                      SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " node name: " << node.getNodeKind().asStringRef()
                 << '\n';
#endif
    if (!attr) {
        return applyTranspilationToNode(node, stage);
    }

    return applyTranspilationToAttrNode(*attr, node, stage);
}

HandleResult applyTranspilationToNodes(const TranspilationNodes& nodes, SessionStage& stage) {
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    for (const auto& tnode : nodes) {
        // set appropriate parsed KernelInfo and LoopInfo as active for current node
        sema.setParsedKernelInfo(tnode.ki);
        sema.setLoopInfo(tnode.li);
        auto result = applyTranspilationToNode(tnode.attr, tnode.node, stage);
        if (!result) {
            return tl::make_unexpected(result.error());
        }
    }

    return {};
}

// remove system header to avoid insertion of them during fusion of final transpiled kernel
void removeSystemHeaders(const HeaderDepsInfo& deps, SessionStage& stage) {
    auto& rewriter = stage.getRewriter();
    for (const auto& dep : deps.topLevelDeps) {
        if (!SrcMgr::isSystem(dep.fileType)) {
            continue;
        }
#ifdef OKL_SEMA_DEBUG_LOG
        llvm::outs() << "remove system include " << dep.relativePath << " "
                     << dep.fileName << " \n";
#endif
        rewriter.RemoveText({dep.hashLoc, dep.filenameRange.getEnd()});
    }
}

// gather all transpiled files: main input and affected header and also header with removed system
// includes
TransformedFiles gatherTransformedFiles(SessionStage& stage) {
    auto inputs = stage.getRewriterResultForHeaders();
    inputs.fileMap.merge(stage.getSession().normalizedHeaders.fileMap);
    inputs.fileMap["okl_kernel.cpp"] = stage.getRewriterResultForMainFile();
    return inputs;
}

tl::expected<std::string, Error> preprocessedInputs(const TransformedFiles& inputs,
                                                    SessionStage& stage) {
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

    invocation->getFrontendOpts().Inputs.push_back(
        FrontendInputFile("okl_kernel.cpp", Language::CXX));
    invocation->getFrontendOpts().ProgramAction = frontend::PrintPreprocessedInput;
    invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";

    CompilerInstance compiler;
    compiler.setInvocation(std::move(invocation));
    compiler.createDiagnostics();
    compiler.createFileManager(
        makeOverlayFs(stage.getCompiler().getFileManager().getVirtualFileSystemPtr(), inputs));

    // XXX clang PrintPreprocessedInput action currently can provide output in two ways:
    //     - print it into STDOUT
    //     - write to the file
    //     second addional option  is used to dump output into FS and then
    //     read/delete it
    //
    if (!ExecuteCompilerInvocation(&compiler)) {
        std::filesystem::remove(outputFileName);
        return tl::make_unexpected(
            Error{{}, "failed to make preprocessing okl_kernel.cpp: "});
    }

    auto preprocessedAndFused = util::readFileAsStr(outputFileName);
    if (!preprocessedAndFused) {
        return tl::make_unexpected(Error{{}, "failed to read file " + outputFileName});
    }
    std::filesystem::remove(outputFileName);

    return preprocessedAndFused.value();
}

std::string restoreSystemAndBackendHeaders(std::string& input, const HeaderDepsInfo& deps) {
    // insert backend specific headers
    for (const auto& dep : deps.backendDeps) {
        input.insert(0, dep);
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

tl::expected<std::string, Error> fuseIncludeDeps(const HeaderDepsInfo& deps, SessionStage& stage) {
    removeSystemHeaders(deps, stage);

    auto inputs = gatherTransformedFiles(stage);

    auto preprocessedResult = preprocessedInputs(inputs, stage);
    if (!preprocessedResult) {
        return preprocessedResult;
    }

    auto finalTranspiledKernel = restoreSystemAndBackendHeaders(preprocessedResult.value(), deps);

    return std::move(finalTranspiledKernel);
}
}  // namespace

namespace oklt {
tl::expected<std::string, Error> generateTranspiledCode(SessionStage& stage) {
    const auto& nodes = stage.tryEmplaceUserCtx<TranspilationNodes>();
    auto result = applyTranspilationToNodes(nodes, stage);
    if (!result) {
        return tl::make_unexpected(std::move(result.error()));
    }

    const auto& deps = stage.tryEmplaceUserCtx<HeaderDepsInfo>();
    auto finalResult = fuseIncludeDeps(deps, stage);
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

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "Program metadata: " << kernelMetaData << "\n";
    util::writeFileAsStr("metadata.json", kernelMetaData);
#endif

    return kernelMetaData;
}
}  // namespace oklt
