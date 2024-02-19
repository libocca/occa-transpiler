#pragma once

#include <map>
#include "oklt/core/attribute_manager/attr_decl_handler.h"
#include "oklt/core/attribute_manager/attr_stmt_handler.h"

namespace oklt {

class CommonAttributeMap {
   public:
    using DeclHandlers = std::map<std::string, AttrDeclHandler>;
    using StmtHandlers = std::map<std::string, AttrStmtHandler>;

    CommonAttributeMap() = default;
    ~CommonAttributeMap() = default;

    bool registerHandler(std::string name, AttrDeclHandler handler);
    bool registerHandler(std::string name, AttrStmtHandler handler);

    bool handleAttr(const clang::Attr* attr, const clang::Decl* decl, SessionStage& stage);
    bool handleAttr(const clang::Attr* attr, const clang::Stmt* stmt, SessionStage& stage);

    bool hasAttrHandler(const std::string& name);

   private:
    DeclHandlers _declHandlers;
    StmtHandlers _stmtHandlers;
};

}  // namespace oklt
