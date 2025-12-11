#pragma once

#include <string>

#include "kotelnikova_a_num_sent_in_line/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kotelnikova_a_num_sent_in_line {

class KotelnikovaANumSentInLineMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KotelnikovaANumSentInLineMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int CountLocalSentences(const std::string &text, int start, int end, int total_length);
  static bool BorderControl(const std::string &text, int start);
  static void ProcessingPart(const std::string &text, int start, int end, int &local_count, bool &local_in_sentence);
};

}  // namespace kotelnikova_a_num_sent_in_line
