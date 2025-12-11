#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "tabalaev_a_elem_mat_min/common/include/common.hpp"
#include "tabalaev_a_elem_mat_min/mpi/include/ops_mpi.hpp"
#include "tabalaev_a_elem_mat_min/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tabalaev_a_elem_mat_min {

class TabalaevAElemMatMinPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    size_t rows = 5000;
    size_t columns = 5000;
    std::vector<int> matrix(rows * columns);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < columns; j++) {
        int index = static_cast<int>((i * columns) + j);
        matrix[index] = static_cast<int>((i * i) + j);
      }
    }
    matrix[(rows * columns) / 2] = expected_minik_;
    input_data_ = std::make_tuple(rows, columns, matrix);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_minik_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  int expected_minik_ = -1000000;
};

TEST_P(TabalaevAElemMatMinPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TabalaevAElemMatMinMPI, TabalaevAElemMatMinSEQ>(
    PPC_SETTINGS_tabalaev_a_elem_mat_min);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TabalaevAElemMatMinPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TabalaevAElemMatMinPerfTests, kGtestValues, kPerfTestName);

}  // namespace tabalaev_a_elem_mat_min
