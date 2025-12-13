#pragma once

#include <string>

#include "makoveeva_s_number_of_sentence/common/include/common.hpp"
#include "task/include/task.hpp"

namespace makoveeva_s_number_of_sentence {

class SentencesCounterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SentencesCounterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int ProcessTextSegment(const std::string &text_segment, char previous_char);
  static int SkipRepeatedPunctuation(const std::string &text, int current_position);
  static bool IsSentenceEnding(char character);
};

}  // namespace makoveeva_s_number_of_sentence
