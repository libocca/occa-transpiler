#pragma once

namespace oklt {

template<typename ... Ts>
struct Overload : Ts ... {
  using Ts::operator() ...;
};
template<class... Ts> Overload(Ts...) -> Overload<Ts...>;
}
