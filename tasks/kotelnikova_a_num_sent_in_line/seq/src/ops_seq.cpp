#include "kotelnikova_a_num_sent_in_line/seq/include/ops_seq.hpp"

#include <cctype>
#include <cstddef>
#include <string>

#include "kotelnikova_a_num_sent_in_line/common/include/common.hpp"

namespace kotelnikova_a_num_sent_in_line {

KotelnikovaANumSentInLineSEQ::KotelnikovaANumSentInLineSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KotelnikovaANumSentInLineSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool KotelnikovaANumSentInLineSEQ::PreProcessingImpl() {
  return true;
}

bool KotelnikovaANumSentInLineSEQ::RunImpl() {
  const std::string &text = GetInput();

  std::size_t sentence_count = 0;
  bool in_sentence = false;

  for (std::size_t i = 0; i < text.length(); ++i) {
    char current_symb = text[i];

    if (current_symb == '.' || current_symb == '!' || current_symb == '?') {
      if (in_sentence) {
        sentence_count++;
        in_sentence = false;
      }
    } else if (std::isalnum(static_cast<unsigned char>(current_symb)) != 0) {
      in_sentence = true;
    }
  }

  if (in_sentence) {
    sentence_count++;
  }

  GetOutput() = sentence_count;
  return true;
}

bool KotelnikovaANumSentInLineSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kotelnikova_a_num_sent_in_line
