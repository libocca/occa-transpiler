#pragma once

#include "core/attribute_manager/attr_decl_handler.h"
#include "core/attribute_manager/attr_stmt_handler.h"

#include <map>
namespace oklt {

class CommonAttributeMap {
   public:
    using DeclHandlers = std::map<std::string, AttrDeclHandler>;
    using StmtHandlers = std::map<std::string, AttrStmtHandler>;

    CommonAttributeMap() = default;
    ~CommonAttributeMap() = default;

    bool registerHandler(std::string name, AttrDeclHandler handler);
    bool registerHandler(std::string name, AttrStmtHandler handler);

    tl::expected<std::any, Error> handleAttr(const clang::Attr* attr,
                                             const clang::Decl* decl,
                                             const std::any* params,
                                             SessionStage& stage);
    tl::expected<std::any, Error> handleAttr(const clang::Attr* attr,
                                             const clang::Stmt* stmt,
                                             const std::any* params,
                                             SessionStage& stage);

    bool hasAttrHandler(const std::string& name);

   private:
    DeclHandlers _declHandlers;
    StmtHandlers _stmtHandlers;
};

}  // namespace oklt
