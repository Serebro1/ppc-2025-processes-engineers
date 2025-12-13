#include <gtest/gtest.h>

#include <utility>

#include "shemetov_d_find_error_vec/common/include/common.hpp"
#include "shemetov_d_find_error_vec/mpi/include/ops_mpi.hpp"
#include "shemetov_d_find_error_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shemetov_d_find_error_vec {

class ShemetovDFindErrorVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    constexpr int kVectorSize = 90'000'000;
    inputData_.resize(kVectorSize);

    for (int i = 0; i < kVectorSize; ++i) {
      auto base = static_cast<double>(i % 1000);
      double offset = ((i & 1) != 0) ? -1.0 : 1.0;
      inputData_[i] = base + offset;
    }
  }

  bool CheckTestOutputData(OutType &output) final {
    return (output >= 0) && std::cmp_less_equal(output, inputData_.size());
  }

  InType GetTestInputData() final {
    return inputData_;
  }

 private:
  InType inputData_;
};

TEST_P(ShemetovDFindErrorVecPerfTests, RunPerformanceModes) {
  ExecuteTest(GetParam());
}

static const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShemetovDFindErrorVecMPI, ShemetovDFindErrorVecSEQ>(
        PPC_SETTINGS_shemetov_d_find_error_vec);

static const auto kPerfGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

static const auto kPerfTestName = ShemetovDFindErrorVecPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfTests, ShemetovDFindErrorVecPerfTests, kPerfGtestValues, kPerfTestName);

}  // namespace shemetov_d_find_error_vec
