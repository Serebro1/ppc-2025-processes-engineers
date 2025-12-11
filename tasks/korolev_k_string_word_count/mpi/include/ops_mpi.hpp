#pragma once

#include "korolev_k_string_word_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace korolev_k_string_word_count {

class KorolevKStringWordCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KorolevKStringWordCountMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace korolev_k_string_word_count
