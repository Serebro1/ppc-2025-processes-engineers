#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yakimov_i_max_values_in_matrix_rows/common/include/common.hpp"
#include "yakimov_i_max_values_in_matrix_rows/mpi/include/ops_mpi.hpp"
#include "yakimov_i_max_values_in_matrix_rows/seq/include/ops_seq.hpp"

namespace yakimov_i_max_values_in_matrix_rows {

class YakimovIMaxValuesInMatrixRowsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    return output_data != 0;
  }

  InType GetTestInputData() final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    return std::get<0>(params);
  }
};

namespace {

TEST_P(YakimovIMaxValuesInMatrixRowsFuncTests, MaxValuesInRows) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kAllTestParam = {std::make_tuple(1, "1"),
                                                std::make_tuple(2, "2"),
                                                std::make_tuple(3, "3"),
                                                std::make_tuple(4, "4"),
                                                std::make_tuple(5, "5"),
                                                std::make_tuple(31, "edge_1x1"),
                                                std::make_tuple(32, "edge_1x100"),
                                                std::make_tuple(33, "edge_100x1"),
                                                std::make_tuple(34, "edge_small"),
                                                std::make_tuple(35, "edge_negative"),
                                                std::make_tuple(36, "edge_large_vals")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<YakimovIMaxValuesInMatrixRowsMPI, InType>(
                                               kAllTestParam, PPC_SETTINGS_yakimov_i_max_values_in_matrix_rows),
                                           ppc::util::AddFuncTask<YakimovIMaxValuesInMatrixRowsSEQ, InType>(
                                               kAllTestParam, PPC_SETTINGS_yakimov_i_max_values_in_matrix_rows));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    YakimovIMaxValuesInMatrixRowsFuncTests::PrintFuncTestName<YakimovIMaxValuesInMatrixRowsFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixTests, YakimovIMaxValuesInMatrixRowsFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace yakimov_i_max_values_in_matrix_rows
