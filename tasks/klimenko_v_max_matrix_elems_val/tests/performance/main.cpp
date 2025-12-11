#include <gtest/gtest.h>

#include "klimenko_v_max_matrix_elems_val/common/include/common.hpp"
#include "klimenko_v_max_matrix_elems_val/mpi/include/ops_mpi.hpp"
#include "klimenko_v_max_matrix_elems_val/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace klimenko_v_max_matrix_elems_val {

class KlimenkoVMaxMatrixElemsValPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int n_ = 5000;
  InType input_data_;
  OutType expected_max_ = 0;

  void SetUp() override {
    input_data_.resize(n_);
    int val = 1;
    for (int i = 0; i < n_; i++) {
      input_data_[i].resize(n_);
      for (int j = 0; j < n_; j++) {
        input_data_[i][j] = val++;
      }
    }
    expected_max_ = n_ * n_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_max_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KlimenkoVMaxMatrixElemsValPerfTests, FindMatrixMax) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KlimenkoVMaxMatrixElemsValMPI, KlimenkoVMaxMatrixElemsValSEQ>(
        PPC_SETTINGS_klimenko_v_max_matrix_elems_val);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = KlimenkoVMaxMatrixElemsValPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MatrixTestsPerf, KlimenkoVMaxMatrixElemsValPerfTests, kGtestValues, kPerfTestName);
}  // namespace klimenko_v_max_matrix_elems_val
