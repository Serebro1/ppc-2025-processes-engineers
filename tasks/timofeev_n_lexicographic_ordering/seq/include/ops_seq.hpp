#pragma once

#include "task/include/task.hpp"
#include "timofeev_n_lexicographic_ordering/common/include/common.hpp"

namespace timofeev_n_lexicographic_ordering {

class TimofeevNLexicographicOrderingSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TimofeevNLexicographicOrderingSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace timofeev_n_lexicographic_ordering
