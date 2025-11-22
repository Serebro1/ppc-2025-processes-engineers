#pragma once

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "task/include/task.hpp"
namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit OtcheskovSElemVecAvgMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int proc_rank_;
  int proc_num_;
  std::vector<int> local_data_;
  std::vector<int> displacements_;
  std::vector<int> counts_;
};

}  // namespace otcheskov_s_elem_vec_avg
