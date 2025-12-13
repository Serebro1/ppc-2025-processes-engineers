#pragma once

#include "makoveeva_s_number_of_sentence/common/include/common.hpp"
#include "task/include/task.hpp"

namespace makoveeva_s_number_of_sentence {

class SentencesCounterSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SentencesCounterSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace makoveeva_s_number_of_sentence
