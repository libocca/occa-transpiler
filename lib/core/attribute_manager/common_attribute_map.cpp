#include "oklt/core/attribute_manager/common_attribute_map.h"

namespace oklt {
using namespace clang;

void CommonAttributeMap::registerHandler(std::string &&name, AttrDeclHandler &&handler) {
  _declHandlers.insert(std::make_pair(std::move(name), std::move(handler)));
}

void CommonAttributeMap::registerHandler(std::string &&name, AttrStmtHandler &&handler) {
   _stmtHandlers.insert(std::make_pair(std::move(name), std::move(handler)));
}

bool CommonAttributeMap::handleAttr(const Attr *attr,
                                    const Decl *decl,
                                    TranspileSession &session)
{
  std::string name = attr->getNormalizedFullName();
  auto it = _declHandlers.find(name);
  if(it != _declHandlers.end()) {
    return it->second.handle(attr, decl, session);
  }
  return false;
}

bool CommonAttributeMap::handleAttr(const Attr *attr,
                                    const Stmt *stmt,
                                    TranspileSession &session)
{
  std::string name = attr->getNormalizedFullName();
  auto it = _stmtHandlers.find(name);
  if(it != _stmtHandlers.end()) {
    return it->second.handle(attr, stmt, session);
  }
  return false;
}

bool CommonAttributeMap::hasAttrHandler(const std::string &name)
{
  auto declIt = _declHandlers.find(name);
  if(declIt != _declHandlers.cend()) {
    return true;
  }
  auto stmtIt = _stmtHandlers.find(name);
  return stmtIt != _stmtHandlers.cend();
}

}
