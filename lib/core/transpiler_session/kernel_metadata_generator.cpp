#include <oklt/util/io_helper.h>

#include "core/sema/okl_sema_ctx.h"

#include "core/transpiler_session/kernel_metadata_generator.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt{
struct Error;
class SessionStage;

tl::expected<std::string, Error> generateKernelMetaData(SessionStage& stage) {
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
}
