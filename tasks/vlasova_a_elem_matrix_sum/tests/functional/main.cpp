#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"
#include "vlasova_a_elem_matrix_sum/mpi/include/ops_mpi.hpp"
#include "vlasova_a_elem_matrix_sum/seq/include/ops_seq.hpp"

namespace vlasova_a_elem_matrix_sum {

class VlasovaAElemMatrixSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string matrix_name = std::get<1>(test_params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vlasova_a_elem_matrix_sum, matrix_name + ".txt");
    std::ifstream file(abs_path);

    int rows = 0;
    int cols = 0;

    file >> rows >> cols;

    std::vector<int> matrix_data(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        file >> matrix_data[(i * cols) + j];
      }
    }

    input_data_ = std::make_tuple(matrix_data, rows, cols);

    expected_result_.resize(rows);
    for (int i = 0; i < rows; ++i) {
      int row_sum = 0;
      for (int j = 0; j < cols; ++j) {
        row_sum += matrix_data[(i * cols) + j];
      }
      expected_result_[i] = row_sum;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_;
};

namespace {

TEST_P(VlasovaAElemMatrixSumFuncTests, MatrixRowSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "matrix1"), std::make_tuple(2, "matrix2"),
                                            std::make_tuple(3, "matrix3")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VlasovaAElemMatrixSumMPI, InType>(kTestParam, PPC_SETTINGS_vlasova_a_elem_matrix_sum),
    ppc::util::AddFuncTask<VlasovaAElemMatrixSumSEQ, InType>(kTestParam, PPC_SETTINGS_vlasova_a_elem_matrix_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VlasovaAElemMatrixSumFuncTests::PrintFuncTestName<VlasovaAElemMatrixSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixRowSum, VlasovaAElemMatrixSumFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace vlasova_a_elem_matrix_sum
