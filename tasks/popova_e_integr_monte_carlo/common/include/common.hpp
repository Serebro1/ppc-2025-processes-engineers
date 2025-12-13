#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace popova_e_integr_monte_carlo {

enum class FuncType : std::uint8_t {
  kLinearFunc = 0,     // 2x + 7
  kQuadraticFunc = 1,  // 5x - 3x^2 + 7
  kCubicFunc = 2,      // x^3 - 4x
  kCosFunc = 3,        // cos(2x)
  kExpFunc = 4         // 2x * exp(-2x) + 4
};

class FunctionPair {
 public:
  static double Function(FuncType func_id, double x) {
    switch (func_id) {
      case FuncType::kLinearFunc:
        return ((2.0 * x) + 7.0);
      case FuncType::kQuadraticFunc:
        return ((5.0 * x) - (3.0 * x * x) + 7.0);
      case FuncType::kCubicFunc:
        return ((x * x * x) - (4.0 * x));
      case FuncType::kCosFunc:
        return (std::cos(2.0 * x));
      case FuncType::kExpFunc:
        return ((2.0 * x * std::exp(-2.0 * x)) + 4.0);
      default:
        return 0.0;
    }
  }

  static double Integral(FuncType func_id, double x) {
    switch (func_id) {
      case FuncType::kLinearFunc:
        return ((x * x) + (7.0 * x));
      case FuncType::kQuadraticFunc:
        return ((2.5 * x * x) - (x * x * x) + (7.0 * x));
      case FuncType::kCubicFunc:
        return ((0.25 * x * x * x * x) - (2.0 * x * x));
      case FuncType::kCosFunc:
        return (0.5 * std::sin(2.0 * x));
      case FuncType::kExpFunc:
        return ((-(x + 0.5) * std::exp(-2.0 * x)) + (4.0 * x));
      default:
        return 0.0;
    }
  }
};

using InType = std::tuple<double, double, int, FuncType>;
using OutType = double;
using TestType = std::tuple<std::tuple<double, double, int, FuncType>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace popova_e_integr_monte_carlo
