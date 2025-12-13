#include <gtest/gtest.h>

#include <algorithm>  // для std::max
#include <iostream>   // для std::cout
#include <limits>     // для std::numeric_limits

#include "ivanova_p_max_matrix/common/include/common.hpp"
#include "ivanova_p_max_matrix/data/matrix_generator.hpp"
#include "ivanova_p_max_matrix/mpi/include/ops_mpi.hpp"
#include "ivanova_p_max_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ivanova_p_max_matrix {

class IvanovaPMaxMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    GenerateTestMatrix();
  }

  void GenerateTestMatrix() {
    const int perf_matrix_size = 10000;
    input_data_ = data::MatrixGenerator::GenerateSquareMatrixWithKnownMax(perf_matrix_size);
    expected_max_ = perf_matrix_size;

    int actual_max = std::numeric_limits<int>::min();
    for (const auto &row : input_data_) {
      for (int val : row) {
        actual_max = std::max(val, actual_max);
      }
    }

    std::cout << "Perf Test: Matrix " << perf_matrix_size << "x" << perf_matrix_size
              << " - Expected max: " << expected_max_ << ", Actual max: " << actual_max << '\n';
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data == expected_max_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  int expected_max_ = 0;
};

TEST_P(IvanovaPMaxMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, IvanovaPMaxMatrixMPI, IvanovaPMaxMatrixSEQ>(PPC_SETTINGS_ivanova_p_max_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IvanovaPMaxMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, IvanovaPMaxMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace ivanova_p_max_matrix
