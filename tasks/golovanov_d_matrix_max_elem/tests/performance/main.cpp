#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "golovanov_d_matrix_max_elem//common/include/common.hpp"
#include "golovanov_d_matrix_max_elem//mpi/include/ops_mpi.hpp"
#include "golovanov_d_matrix_max_elem//seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace golovanov_d_matrix_max_elem {

class GolovanovDMatrixMaxElemPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  double maximum_ = 100000;

 public:
  void SetUp() override {
    std::vector<double> tmp_vector(0);
    int n = 2000;
    int m = 2000;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        double value = i + j;
        tmp_vector.push_back(value);
      }
    }
    tmp_vector[10] = maximum_;
    input_data_ = std::tuple<int, int, std::vector<double>>(n, m, tmp_vector);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == maximum_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GolovanovDMatrixMaxElemPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, GolovanovDMatrixMaxElemMPI, GolovanovDMatrixMaxElemSEQ>(
    PPC_SETTINGS_golovanov_d_matrix_max_elem);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GolovanovDMatrixMaxElemPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MatrixMaxElemPerTests, GolovanovDMatrixMaxElemPerfTest, kGtestValues, kPerfTestName);

}  // namespace golovanov_d_matrix_max_elem
