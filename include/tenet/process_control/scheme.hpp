#pragma once
// include/tenet/process_control/scheme.hpp
//
// Tag types for compile-time dispatch of sweep schemes and directions.
// See docs/C++重构设计方案.md §4.4.

namespace tenet {

// Sweep update schemes
struct SingleSite {};
struct DoubleSite {};

// Sweep directions
struct L2R {};   // Left-to-Right
struct R2L {};   // Right-to-Left

} // namespace tenet
