#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "shemetov_d_find_error_vec/common/include/common.hpp"
#include "shemetov_d_find_error_vec/mpi/include/ops_mpi.hpp"
#include "shemetov_d_find_error_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shemetov_d_find_error_vec {

class ShemetovDFindErrorVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  // Имя теста = строковый тег из TestType
  static std::string PrintTestParam(const TestType &param) {
    return std::get<2>(param);
  }

 protected:
  void SetUp() override {
    const auto &test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    inputData_ = std::get<0>(test_params);
  }

  bool CheckTestOutputData(OutType &output) final {
    const double eps = 1e-10;
    int ref = 0;

    for (size_t i = 0; i + 1 < inputData_.size(); i += 1) {
      if (inputData_[i] - inputData_[i + 1] > eps) {
        ref += 1;
      }
    }

    return output == ref;
  }

  InType GetTestInputData() final {
    return inputData_;
  }

 private:
  InType inputData_;
};

static const std::array<TestType, 14> kTestCases = {
    std::make_tuple(std::vector<double>{}, 0, "empty"),
    std::make_tuple(std::vector<double>{1.0}, 0, "single"),

    std::make_tuple(std::vector<double>{1.0, 2.0}, 0, "simple_no_violation"),
    std::make_tuple(std::vector<double>{2.0, 1.0}, 1, "simple_violation"),

    std::make_tuple(std::vector<double>{1, 2, 3, 4, 5}, 0, "sorted_inc"),
    std::make_tuple(std::vector<double>{5, 4, 3, 2, 1}, 4, "sorted_dec"),

    std::make_tuple(std::vector<double>{1, 3, 2, 5, 4}, 2, "mixed_pattern"),
    std::make_tuple(std::vector<double>{3, 1, 2}, 1, "three_numbers"),

    std::make_tuple(std::vector<double>{1, 1, 1, 1}, 0, "same_values"),
    std::make_tuple(std::vector<double>{1, 2, 2, 3, 3}, 0, "non_decreasing"),

    std::make_tuple(std::vector<double>{1, 1 + 1e-11, 1 + 1e-9}, 0, "precision_safe"),
    std::make_tuple(std::vector<double>{1, 1 - 1e-8, 1 - 2e-8}, 2, "precision_falling"),

    std::make_tuple(std::vector<double>{10, 1, 9, 2, 8, 3, 7}, 6, "zigzag_desc"),
    std::make_tuple(std::vector<double>(1000, 1.0), 0, "large_flat")};

TEST_P(ShemetovDFindErrorVecFuncTests, Correctness) {
  ExecuteTest(GetParam());
}

static const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<ShemetovDFindErrorVecMPI, InType>(kTestCases, PPC_SETTINGS_shemetov_d_find_error_vec),
    ppc::util::AddFuncTask<ShemetovDFindErrorVecSEQ, InType>(kTestCases, PPC_SETTINGS_shemetov_d_find_error_vec));

INSTANTIATE_TEST_SUITE_P(Functional, ShemetovDFindErrorVecFuncTests, ppc::util::ExpandToValues(kTaskList),
                         ShemetovDFindErrorVecFuncTests::PrintFuncTestName<ShemetovDFindErrorVecFuncTests>);

}  // namespace shemetov_d_find_error_vec
