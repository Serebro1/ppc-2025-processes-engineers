#pragma once

#include "alekseev_a_min_dist_neigh_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace alekseev_a_min_dist_neigh_elem_vec {

class AlekseevAMinDistNeighElemVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit AlekseevAMinDistNeighElemVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace alekseev_a_min_dist_neigh_elem_vec
