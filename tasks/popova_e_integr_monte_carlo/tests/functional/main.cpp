#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "popova_e_integr_monte_carlo/common/include/common.hpp"
#include "popova_e_integr_monte_carlo/mpi/include/ops_mpi.hpp"
#include "popova_e_integr_monte_carlo/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace popova_e_integr_monte_carlo {

class PopovaERunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &[a, b, n, func_id] = input_data_;

    double exp_integral = 0.0;
    exp_integral = FunctionPair::Integral(func_id, b) - FunctionPair::Integral(func_id, a);

    double sredn = exp_integral / (b - a);
    double std_dev = (b - a) / std::sqrt(n) * std::max(std::abs(sredn), 1.0);
    double epsilon = std::max(10.0 * std_dev, 1e-2);
    return std::abs(output_data - exp_integral) <= epsilon;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(PopovaERunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    {std::make_tuple(std::make_tuple(0.0, 1.0, 1000, FuncType::kLinearFunc), "kLinearFunc_test1"),
     std::make_tuple(std::make_tuple(-1.0, 5.0, 80000, FuncType::kLinearFunc), "kLinearFunc_test2"),

     std::make_tuple(std::make_tuple(0.0, 2.0, 1000, FuncType::kQuadraticFunc), "kQuadraticFunc_test1"),
     std::make_tuple(std::make_tuple(-1.0, 5.0, 45000, FuncType::kQuadraticFunc), "kQuadraticFunc_test2"),

     std::make_tuple(std::make_tuple(1.0, 3.0, 500, FuncType::kCubicFunc), "kCubicFunc_test1"),
     std::make_tuple(std::make_tuple(4.5, 5.0, 100, FuncType::kCubicFunc), "kCubicFunc_test2"),

     std::make_tuple(std::make_tuple(0.0, 1.0, 2000, FuncType::kCosFunc), "kCosFunc_test1"),
     std::make_tuple(std::make_tuple(4.5, 5.0, 8000, FuncType::kCosFunc), "kCosFunc_test2"),

     std::make_tuple(std::make_tuple(0.0, 1.0, 1000, FuncType::kExpFunc), "kExpFunc_test1"),
     std::make_tuple(std::make_tuple(-1.0, 5.0, 80000, FuncType::kExpFunc), "kExpFunc_test2")}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PopovaEIntegrMonteCarloMPI, InType>(kTestParam, PPC_SETTINGS_popova_e_integr_monte_carlo),
    ppc::util::AddFuncTask<PopovaEIntegrMonteCarloSEQ, InType>(kTestParam, PPC_SETTINGS_popova_e_integr_monte_carlo));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PopovaERunFuncTestsProcesses::PrintFuncTestName<PopovaERunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MonteCarloTests, PopovaERunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace popova_e_integr_monte_carlo
