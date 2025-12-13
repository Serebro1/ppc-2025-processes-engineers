#include <gtest/gtest.h>

#include <cmath>
#include <tuple>

#include "iskhakov_d_trapezoidal_integration/common/include/common.hpp"
#include "iskhakov_d_trapezoidal_integration/mpi/include/ops_mpi.hpp"
#include "iskhakov_d_trapezoidal_integration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace iskhakov_d_trapezoidal_integration {

class IskhakovDTrapezoidalIntegrationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = std::make_tuple(0.0, 1.0, InFunction, 10000000);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr double kExpectedResult = 1.8600;
    constexpr double kRelativeTolerance = 0.01;

    double relative_error = std::abs(output_data - kExpectedResult) / std::abs(kExpectedResult);
    return relative_error < kRelativeTolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;

  static double InFunction(double x) {
    return ((x * x * x) * std::sin(x)) + (2.0 * std::cos(x));
  }
};

TEST_P(IskhakovDTrapezoidalIntegrationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, IskhakovDTrapezoidalIntegrationMPI, IskhakovDTrapezoidalIntegrationSEQ>(
        PPC_SETTINGS_iskhakov_d_trapezoidal_integration);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IskhakovDTrapezoidalIntegrationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IskhakovDTrapezoidalIntegrationPerfTests, kGtestValues, kPerfTestName);

}  // namespace iskhakov_d_trapezoidal_integration
