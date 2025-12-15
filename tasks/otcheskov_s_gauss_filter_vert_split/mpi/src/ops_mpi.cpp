#include "otcheskov_s_gauss_filter_vert_split/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "ops_mpi.hpp"
#include "otcheskov_s_gauss_filter_vert_split/common/include/common.hpp"

namespace otcheskov_s_elem_vec_avg {

otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::OtcheskovSGaussFilterVertSplitMPI(
    const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::ValidationImpl() {
  bool is_valid = false;
  if (proc_rank_ == 0) {
    return true;
  }
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

int otcheskov_s_gauss_filter_vert_split::OtcheskovSGaussFilterVertSplitMPI::GetIndex(int row, int col,
                                                                                     int channel) const {
  return (row * GetInput().width + col) * GetInput().channels + channel;
}
}  // namespace otcheskov_s_elem_vec_avg
