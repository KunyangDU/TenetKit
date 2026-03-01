#pragma once
// include/tenet/utils/io.hpp
//
// HDF5 read / write utilities for MPS and result data.
// Requires HDF5 C++ library (optional dependency).

#ifdef TENET_HAS_HDF5
#include <H5Cpp.h>
#endif

#include <string>
#include <vector>

namespace tenet::io {

// ── HDF5 helpers (only available when HDF5 is linked) ───────────────────────

#ifdef TENET_HAS_HDF5

void write_vector(H5::Group& grp, const std::string& name,
                  const std::vector<double>& data);

std::vector<double> read_vector(const H5::Group& grp, const std::string& name);

#endif // TENET_HAS_HDF5

} // namespace tenet::io
