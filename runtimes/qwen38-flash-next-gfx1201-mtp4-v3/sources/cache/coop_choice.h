#pragma once
#include <cstdint>
#include <climits>
#ifndef FMD_HOST_DEVICE
#define FMD_HOST_DEVICE
#endif
namespace full_mutable_device {
struct CoopChoice { int64_t clock; int slot; int kind; };
FMD_HOST_DEVICE inline CoopChoice better_choice(CoopChoice a, CoopChoice b) {
  if (b.kind < a.kind || (b.kind == a.kind &&
      (b.clock < a.clock || (b.clock == a.clock && b.slot < a.slot)))) return b;
  return a;
}
}
