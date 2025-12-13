#include "makoveeva_s_number_of_sentence/seq/include/ops_seq.hpp"

#include <string>

#include "makoveeva_s_number_of_sentence/common/include/common.hpp"

namespace makoveeva_s_number_of_sentence {

SentencesCounterSEQ::SentencesCounterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SentencesCounterSEQ::ValidationImpl() {
  return true;
}

bool SentencesCounterSEQ::PreProcessingImpl() {
  return true;
}

bool SentencesCounterSEQ::RunImpl() {
  const std::string &text = GetInput();
  int sentence_count = 0;
  bool in_sentence_end = false;

  for (char c : text) {
    if (c == '.' || c == '!' || c == '?') {
      if (!in_sentence_end) {
        sentence_count++;
        in_sentence_end = true;
      }
    } else {
      in_sentence_end = false;
    }
  }

  GetOutput() = sentence_count;
  return true;
}

bool SentencesCounterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace makoveeva_s_number_of_sentence
