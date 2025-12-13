#pragma once
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace makoveeva_s_number_of_sentence {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<std::string, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace makoveeva_s_number_of_sentence
