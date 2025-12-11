#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kotelnikova_a_num_sent_in_line {

using InType = std::string;
using OutType = std::size_t;
using TestType = std::tuple<std::string, std::size_t>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kotelnikova_a_num_sent_in_line
