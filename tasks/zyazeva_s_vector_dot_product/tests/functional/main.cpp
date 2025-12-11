#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zyazeva_s_vector_dot_product/common/include/common.hpp"
#include "zyazeva_s_vector_dot_product/mpi/include/ops_mpi.hpp"
#include "zyazeva_s_vector_dot_product/seq/include/ops_seq.hpp"

namespace zyazeva_s_vector_dot_product {

class ZyazevaRunFuncTestsSEQ : public ppc::util::BaseRunFuncTests<InType, int64_t, TestType> {
 public:
  static auto PrintTestParam(const TestType &test_param) -> std::string {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int64_t test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = {{1, 2, 3}, {4, 5, 6}};
        expected_output_ = 32;
        break;
      case 1:
        input_data_ = {{5}, {10}};
        expected_output_ = 50;
        break;
      case 2:
        input_data_ = {{2, 2, 2}, {3, 3, 3}};
        expected_output_ = 18;
        break;
      case 3:
        input_data_ = {{100, 200}, {300, 400}};
        expected_output_ = 110000;
        break;
      case 4:
        input_data_ = {{}, {}};
        expected_output_ = 0;
        break;
      case 5:
        input_data_ = {{1, 2, 3}};
        expected_output_ = 0;
        break;
      case 6:
        input_data_ = {{1, 2, 3}, {4, 5}};
        expected_output_ = 0;
        break;
      default:
        input_data_ = {{1, 2}, {3, 4}};
        expected_output_ = 11;
        break;
    }
  }

  auto CheckTestOutputData(int64_t &output_data) -> bool final {  // NOLINT
    return (expected_output_ == output_data);
  }

  auto GetTestInputData() -> InType final {
    return input_data_;
  }

 private:
  InType input_data_;
  int64_t expected_output_{};
};

class ZyazevaRunFuncTestsMPI : public ppc::util::BaseRunFuncTests<InType, int64_t, TestType> {
 public:
  static auto PrintTestParam(const TestType &test_param) -> std::string {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int64_t test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = {{1, 2, 3}, {4, 5, 6}};
        expected_output_ = 32;
        break;
      case 1:
        input_data_ = {{5}, {10}};
        expected_output_ = 50;
        break;
      case 2:
        input_data_ = {{2, 2, 2}, {3, 3, 3}};
        expected_output_ = 18;
        break;
      case 3:
        input_data_ = {{100, 200}, {300, 400}};
        expected_output_ = 110000;
        break;
      case 4:
        input_data_ = {{}, {}};
        expected_output_ = 0;
        break;
      case 5:
        input_data_ = {{1, 2, 3}};
        expected_output_ = 0;
        break;
      case 6:
        input_data_ = {};
        expected_output_ = 0;
        break;
      case 7:
        input_data_ = {{1, 2, 3}, {4, 5}};
        expected_output_ = 0;
        break;
      default:
        input_data_ = {{1, 2}, {3, 4}};
        expected_output_ = 11;
        break;
    }
  }

  auto CheckTestOutputData(int64_t &output_data) -> bool final {  // NOLINT
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
      return (expected_output_ == output_data);
    }
    return true;
  }

  auto GetTestInputData() -> InType final {
    return input_data_;
  }

 private:
  InType input_data_;
  int64_t expected_output_{};
};

namespace {

TEST_P(ZyazevaRunFuncTestsSEQ, DotProductTestSEQ) {
  ExecuteTest(GetParam());
}

TEST_P(ZyazevaRunFuncTestsMPI, DotProductTestMPI) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(0, "simple_vectors"), std::make_tuple(1, "single_element"),  std::make_tuple(2, "all_equal"),
    std::make_tuple(3, "large_values"),   std::make_tuple(4, "empty_vectors"),   std::make_tuple(5, "one_vector"),
    std::make_tuple(6, "nothing"),        std::make_tuple(7, "different_lenght")};

const auto kTestTasksListSEQ =
    ppc::util::AddFuncTask<ZyazevaSVecDotProductSEQ, InType>(kTestParam, PPC_SETTINGS_zyazeva_s_vector_dot_product);

const auto kTestTasksListMPI =
    ppc::util::AddFuncTask<ZyazevaSVecDotProductMPI, InType>(kTestParam, PPC_SETTINGS_zyazeva_s_vector_dot_product);

const auto kGtestValuesSEQ = ppc::util::ExpandToValues(kTestTasksListSEQ);
const auto kGtestValuesMPI = ppc::util::ExpandToValues(kTestTasksListMPI);

const auto kPerfTestNameSEQ = ZyazevaRunFuncTestsSEQ::PrintFuncTestName<ZyazevaRunFuncTestsSEQ>;
const auto kPerfTestNameMPI = ZyazevaRunFuncTestsMPI::PrintFuncTestName<ZyazevaRunFuncTestsMPI>;

INSTANTIATE_TEST_SUITE_P(VectorDotProductTestsSEQ, ZyazevaRunFuncTestsSEQ, kGtestValuesSEQ, kPerfTestNameSEQ);

INSTANTIATE_TEST_SUITE_P(VectorDotProductTestsMPI, ZyazevaRunFuncTestsMPI, kGtestValuesMPI, kPerfTestNameMPI);

}  // namespace

}  // namespace zyazeva_s_vector_dot_product
