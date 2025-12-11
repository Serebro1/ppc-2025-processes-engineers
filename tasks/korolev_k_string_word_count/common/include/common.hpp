#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace korolev_k_string_word_count {

// The input is a string; the output is the number of words in that string.
// A "word" is a maximal sequence of non-whitespace characters.
using InType = std::string;
using OutType = int;

// (input_string, expected_word_count) — helpful for tests.
using TestType = std::tuple<std::string, int>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace korolev_k_string_word_count
