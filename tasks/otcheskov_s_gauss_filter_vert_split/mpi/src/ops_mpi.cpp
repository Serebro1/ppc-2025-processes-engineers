#include "otcheskov_s_gauss_filter_vert_split/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"

namespace otcheskov_s_elem_vec_avg {

otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::OtcheskovSGaussFilterVertSplitMPI(
    const InType &in) {}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::ValidationImpl() {
  return tdrue;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::PreProcessingImpl() {
  return true;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::RunImpl() {
  return true;
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg
