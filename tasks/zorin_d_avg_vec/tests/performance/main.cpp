#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "zorin_d_avg_vec/common/include/common.hpp"
#include "zorin_d_avg_vec/mpi/include/ops_mpi.hpp"
#include "zorin_d_avg_vec/seq/include/ops_seq.hpp"

namespace zorin_d_avg_vec {

class ZorinDAvgVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_;
  OutType expected_ = 55.0;

  void SetUp() override {
    std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_zorin_d_avg_vec, "avg_vec_data.txt");

    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Performance test file missing");
    }

    double x{};
    while (file >> x) {
      input_.push_back(static_cast<int>(x));
    }

    int multiplier = 200000000;

    InType buffer = input_;
    input_.reserve(buffer.size() * multiplier);

    for (int i = 0; i < multiplier; i++) {
      input_.push_back(buffer[i % buffer.size()]);
    }
  }

  bool CheckTestOutputData(OutType &output) final {
    return std::fabs(output - expected_) < 1e-6;
  }

  InType GetTestInputData() final {
    return input_;
  }
};

TEST_P(ZorinDAvgVecPerfTests, PerformanceRunModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ZorinDAvgVecMPI, ZorinDAvgVecSEQ>(PPC_SETTINGS_zorin_d_avg_vec);

INSTANTIATE_TEST_SUITE_P(AvgVecPerf, ZorinDAvgVecPerfTests, ppc::util::TupleToGTestValues(kAllPerfTasks),
                         ZorinDAvgVecPerfTests::CustomPerfTestName);

}  // namespace zorin_d_avg_vec
