#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"
#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

namespace zyazeva_s_vector_dot_product {

class ZyazevaSVectorDotProductPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  int64_t expected_result_ = 0;

  void SetUp() override {
    auto result = GenerateLargeVectorsWithResult(60000000);
    input_data_ = result.first;
    expected_result_ = result.second;
  }

  static std::pair<InType, int64_t> GenerateLargeVectorsWithResult(size_t size) {
    std::vector<std::vector<int32_t>> vectors(2);
    vectors[0].resize(size);
    vectors[1].resize(size);
    int64_t dot_product = 0;

    for (size_t i = 0; i < size; ++i) {
      auto a = static_cast<int32_t>(1 + (i % 100));
      auto b = static_cast<int32_t>(1 + ((i * 7) % 100));
      vectors[0][i] = a;
      vectors[1][i] = b;
      dot_product += static_cast<int64_t>(a) * static_cast<int64_t>(b);
    }

    return std::make_pair(vectors, dot_product);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return static_cast<int64_t>(output_data) == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ZyazevaSVectorDotProductPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ZyazevaSVecDotProductMPI, ZyazevaSVecDotProductSEQ>(
    PPC_SETTINGS_zyazeva_s_vector_dot_product);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZyazevaSVectorDotProductPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZyazevaSVectorDotProductPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace zyazeva_s_vector_dot_product
