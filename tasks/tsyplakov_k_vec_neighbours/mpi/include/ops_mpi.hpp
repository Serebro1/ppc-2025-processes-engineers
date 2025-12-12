#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"

namespace tsyplakov_k_vec_neighbours {

class TsyplakovKVecNeighboursMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit TsyplakovKVecNeighboursMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> local_vec_;
};

}  // namespace tsyplakov_k_vec_neighbours
