#pragma once

#include <oklt/core/target_backends.h>
#include "core/attribute_manager/attr_decl_handler.h"
#include "core/attribute_manager/attr_stmt_handler.h"
#include "core/attribute_manager/result.h"
#include "util/type_traits.h"

#include <map>
#include <tuple>

namespace oklt {
constexpr size_t N_ARGUMENTS_WITH_PARAMS = 4;
constexpr size_t N_ARGUMENTS_WITHOUT_PARAMS = 3;

class BackendAttributeMap {
   public:
    using KeyType = std::tuple<TargetBackend, std::string>;
    using DeclHandlers = std::map<KeyType, AttrDeclHandler>;
    using StmtHandlers = std::map<KeyType, AttrStmtHandler>;

    BackendAttributeMap() = default;
    ~BackendAttributeMap() = default;

    bool registerHandler(KeyType key, AttrDeclHandler handler);
    bool registerHandler(KeyType key, AttrStmtHandler handler);

    HandleResult handleAttr(SessionStage& stage,
                            const clang::Decl& decl,
                            const clang::Attr& attr,
                            const std::any* params);
    HandleResult handleAttr(SessionStage& stage,
                            const clang::Stmt& stmt,
                            const clang::Attr& attr,
                            const std::any* params);

    bool hasAttrHandler(SessionStage& stage, const std::string& name);

   private:
    DeclHandlers _declHandlers;
    StmtHandlers _stmtHandlers;
};
}  // namespace oklt
