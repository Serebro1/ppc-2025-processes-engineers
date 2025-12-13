#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "baranov_a_sign_alternations/common/include/common.hpp"
#include "baranov_a_sign_alternations/mpi/include/ops_mpi.hpp"
#include "baranov_a_sign_alternations/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace baranov_a_sign_alternations {

class BaranovASignAlternationsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    int size = 10000000;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-100, 100);

    input_data_.resize(size);
    for (int i = 0; i < size; i++) {
      if (i % 3 == 0) {
        input_data_[i] = std::abs(dis(gen)) + 1;
      } else if (i % 3 == 1) {
        input_data_[i] = std::abs(dis(gen)) - 1;
      } else {
        input_data_[i] = 0;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(BaranovASignAlternationsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BaranovASignAlternationsMPI, BaranovASignAlternationsSEQ>(
        PPC_SETTINGS_baranov_a_sign_alternations);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BaranovASignAlternationsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BaranovASignAlternationsPerfTests, kGtestValues, kPerfTestName);

}  // namespace baranov_a_sign_alternations
