#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "kopilov_d_sum_val_col_mat/common/include/common.hpp"
#include "kopilov_d_sum_val_col_mat/mpi/include/ops_mpi.hpp"
#include "kopilov_d_sum_val_col_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kopilov_d_sum_val_col_mat {

class KopilovDSumValColMatPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int k_matrix_size = 4000;
  InType input_data;
  OutType expected_output;

  void SetUp() override {
    input_data.rows = k_matrix_size;
    input_data.cols = k_matrix_size;
    input_data.data.resize(static_cast<std::size_t>(k_matrix_size) * static_cast<std::size_t>(k_matrix_size));

    for (int row = 0; row < k_matrix_size; ++row) {
      for (int col = 0; col < k_matrix_size; ++col) {
        input_data.data[(static_cast<std::size_t>(row) * static_cast<std::size_t>(k_matrix_size)) +
                        static_cast<std::size_t>(col)] = static_cast<double>(row + col);
      }
    }

    expected_output.col_sum.resize(static_cast<std::size_t>(k_matrix_size));
    for (int col = 0; col < k_matrix_size; ++col) {
      double sum = 0.0;
      for (int row = 0; row < k_matrix_size; ++row) {
        sum += static_cast<double>(row + col);
      }
      expected_output.col_sum[static_cast<std::size_t>(col)] = sum;
    }
  }

  InType GetTestInputData() override {
    return input_data;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    if (output_data.col_sum.size() != expected_output.col_sum.size()) {
      return false;
    }
    for (std::size_t i = 0; i < expected_output.col_sum.size(); ++i) {
      if (std::abs(output_data.col_sum[i] - expected_output.col_sum[i]) > 1e-9) {
        return false;
      }
    }
    return true;
  }
};

TEST_P(KopilovDSumValColMatPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KopilovDSumValColMatMPI, KopilovDSumValColMatSEQ>(
    PPC_SETTINGS_kopilov_d_sum_val_col_mat);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = KopilovDSumValColMatPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KopilovDSumValColMatPerfTests, kGtestValues, kPerfTestName);

}  // namespace kopilov_d_sum_val_col_mat
