#pragma once

namespace oklt {

template<class ...SemaProcessors>
struct CompositeSema: SemaProcessors...
{
  using SemaProcessors::beforeTraverse...;
  using SemaProcessors::afterTraverse...;
};

}
