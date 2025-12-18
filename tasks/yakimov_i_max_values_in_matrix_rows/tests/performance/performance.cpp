#include <gtest/gtest.h>

#include <array>
#include <cstddef>

#include "util/include/perf_test_util.hpp"
#include "yakimov_i_max_values_in_matrix_rows/common/include/common.hpp"
#include "yakimov_i_max_values_in_matrix_rows/mpi/include/ops_mpi.hpp"
#include "yakimov_i_max_values_in_matrix_rows/seq/include/ops_seq.hpp"

namespace yakimov_i_max_values_in_matrix_rows {

class YakimovIMaxValuesInMatrixRowsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    return output_data != 0;
  }

  InType GetTestInputData() final {
    static size_t test_index = 0;
    static constexpr std::array<InType, 4> kTestSizes = {27, 28, 29, 30};
    InType result = kTestSizes.at(test_index % kTestSizes.size());
    test_index++;
    return result;
  }
};

TEST_P(YakimovIMaxValuesInMatrixRowsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, YakimovIMaxValuesInMatrixRowsMPI, YakimovIMaxValuesInMatrixRowsSEQ>(
        PPC_SETTINGS_yakimov_i_max_values_in_matrix_rows);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = YakimovIMaxValuesInMatrixRowsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YakimovIMaxValuesInMatrixRowsPerfTests, kGtestValues, kPerfTestName);

}  // namespace yakimov_i_max_values_in_matrix_rows
