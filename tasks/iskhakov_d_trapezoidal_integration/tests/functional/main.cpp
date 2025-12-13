#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <string>
#include <tuple>

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"
#include "iskhakov_d_trapezoidal_integration/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_trapezoidal_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace iskhakov_d_trapezoidal_integration {

class IskhakovDTrapezoidalIntegrationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &[input, expected_result] = test_param;

    double lower_level = std::get<0>(input);
    double top_level = std::get<1>(input);
    int steps = std::get<3>(input);

    int lower_level_int = static_cast<int>(lower_level);
    int top_level_int = static_cast<int>(top_level);

    return "FROM_" + std::to_string(lower_level_int) + "_TO_" + std::to_string(top_level_int) + "_STEPS_" +
           std::to_string(steps);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const auto &[input, expected_result] = params;

    input_data_ = input;
    expected_result_ = expected_result;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::abs(expected_result_) < 1e-12) {
      return std::abs(output_data) < 0.01;
    }
    double relative_error = std::abs(output_data - expected_result_) / std::abs(expected_result_);
    return relative_error < 0.01;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  double expected_result_{};
};

struct Functions {
  static double Original(double x) {
    return ((x * x * x) * std::sin(x)) + (2.0 * std::cos(x));
  }

  static double Quadratic(double x) {
    return x * x;
  }

  static double Cubic(double x) {
    return x * x * x;
  }

  static double Sine(double x) {
    return std::sin(x);
  }

  static double Exponential(double x) {
    return std::exp(x);
  }

  static double Linear(double x) {
    return (2.0 * x) + 1.0;
  }

  static double Constant([[maybe_unused]] double x) {
    return 5.0;
  }

  static double ComplexTrig(double x) {
    return std::sin(x) + std::cos(2.0 * x);
  }

  static double Rational(double x) {
    return 1.0 / (1.0 + (x * x));
  }

  static double Logarithmic(double x) {
    return std::log(1.0 + x);
  }
};

namespace {

InType CreateTestData(double lower_level, double top_level, const std::function<double(double)> &func, int steps) {
  return std::make_tuple(lower_level, top_level, func, steps);
}

TEST_P(IskhakovDTrapezoidalIntegrationFuncTests, TrapezoidalIntegration) {
  ExecuteTest(GetParam());
}

constexpr double kPi = std::numbers::pi;

const std::array<TestType, 15> kTestParam = {
    std::make_tuple(CreateTestData(0.0, 1.0, Functions::Original, 1000), 1.8600),
    std::make_tuple(CreateTestData(0.0, 2.0, Functions::Original, 2000), 5.6100),
    std::make_tuple(CreateTestData(1.0, 3.0, Functions::Original, 3000), 10.2953),

    std::make_tuple(CreateTestData(0.0, 1.0, Functions::Quadratic, 11000), 1.0 / 3.0),
    std::make_tuple(CreateTestData(0.0, 2.0, Functions::Quadratic, 12000), 8.0 / 3.0),
    std::make_tuple(CreateTestData(1.0, 3.0, Functions::Cubic, 13000), 20.0),

    std::make_tuple(CreateTestData(0.0, kPi, Functions::Sine, 14000), 2.0),
    std::make_tuple(CreateTestData(0.0, kPi / 2, Functions::Sine, 15000), 1.0),
    std::make_tuple(CreateTestData(0.0, 2 * kPi, Functions::Sine, 16000), 0.0),

    std::make_tuple(CreateTestData(0.0, 1.0, Functions::Exponential, 17000), std::numbers::e - 1.0),
    std::make_tuple(CreateTestData(1.0, 2.0, Functions::Exponential, 18000),
                    (std::numbers::e * std::numbers::e) - std::numbers::e),

    std::make_tuple(CreateTestData(0.0, 1.0, Functions::Linear, 19000), 2.0),
    std::make_tuple(CreateTestData(0.0, 2.0, Functions::Linear, 20000), 6.0),

    std::make_tuple(CreateTestData(0.0, 5.0, Functions::Constant, 21000), 25.0),

    std::make_tuple(CreateTestData(0.0, kPi, Functions::ComplexTrig, 22000), 2.0)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<IskhakovDTrapezoidalIntegrationMPI, InType>(
                                               kTestParam, PPC_SETTINGS_iskhakov_d_trapezoidal_integration),
                                           ppc::util::AddFuncTask<IskhakovDTrapezoidalIntegrationSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_iskhakov_d_trapezoidal_integration));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    IskhakovDTrapezoidalIntegrationFuncTests::PrintFuncTestName<IskhakovDTrapezoidalIntegrationFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, IskhakovDTrapezoidalIntegrationFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace iskhakov_d_trapezoidal_integration
