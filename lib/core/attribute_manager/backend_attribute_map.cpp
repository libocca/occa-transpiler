#include "oklt/core/attribute_manager/backend_attribute_map.h"
#include "oklt/core/transpile_session/transpile_session.h"

namespace oklt {
using namespace clang;

void BackendAttributeMap::registerHandler(KeyType &&key,
                     AttrDeclHandler &&handler)
{
  _declHandlers.insert(std::make_pair(std::move(key), std::move(handler)));
}

void BackendAttributeMap::registerHandler(KeyType &&key,
                     AttrStmtHandler &&handler)
{
  _stmtHandlers.insert(std::make_pair(std::move(key), std::move(handler)));
}

bool BackendAttributeMap::handleAttr(const Attr *attr,
                                     const Decl *decl,
                                     TranspileSession &session)
{
  std::string name = attr->getNormalizedFullName();
  auto backend = session.getBackend();
  auto it = _declHandlers.find(std::make_tuple(backend, name));
  if(it != _declHandlers.end()) {
    return it->second.handle(attr, decl, session);
  }
  return false;
}

bool BackendAttributeMap::handleAttr(const Attr *attr,
                                     const Stmt *stmt,
                                     TranspileSession &session)
{
  std::string name = attr->getNormalizedFullName();
  auto backend = session.getBackend();
  auto it = _stmtHandlers.find(std::make_tuple(backend, name));
  if(it != _stmtHandlers.end()) {
    return it->second.handle(attr, stmt, session);
  }
  return false;
}

bool BackendAttributeMap::hasAttrHandler(TranspileSession &session,
                                         const std::string &name)
{
  auto key = std::make_tuple(session.getBackend(), name);
  auto declIt = _declHandlers.find(key);
  if(declIt != _declHandlers.cend()) {
    return true;
  }
  auto stmtIt = _stmtHandlers.find(key);
  return stmtIt != _stmtHandlers.cend();
}
}
