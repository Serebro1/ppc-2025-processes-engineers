#include <gtest/gtest.h>

#include <algorithm>  // для std::max
#include <array>
#include <cstddef>   // для std::size_t
#include <iostream>  // для std::cout
#include <limits>    // для std::numeric_limits
#include <string>
#include <tuple>

#include "ivanova_p_max_matrix/common/include/common.hpp"
#include "ivanova_p_max_matrix/data/matrix_generator.hpp"
#include "ivanova_p_max_matrix/mpi/include/ops_mpi.hpp"
#include "ivanova_p_max_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace ivanova_p_max_matrix {

class IvanovaPMaxMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  IvanovaPMaxMatrixFuncTests() = default;

  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    matrix_size_ = std::get<0>(params);
    matrix_type_ = std::get<1>(params);

    if (matrix_type_ == "negative") {
      test_matrix_ = data::MatrixGenerator::GenerateAllNegativeMatrix(matrix_size_, matrix_size_);
      expected_max_ = -1;
    } else {
      test_matrix_ = data::MatrixGenerator::GenerateSquareMatrixWithKnownMax(matrix_size_);
      expected_max_ = matrix_size_;
    }

    int actual_max = std::numeric_limits<int>::min();
    for (const auto &row : test_matrix_) {
      for (int val : row) {
        actual_max = std::max(val, actual_max);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool result = (expected_max_ == output_data);
    if (!result) {
      std::cout << "ERROR: Expected " << expected_max_ << " but got " << output_data << '\n';
    }
    return result;
  }

  InType GetTestInputData() final {
    return test_matrix_;
  }

 private:
  InType test_matrix_;
  int matrix_size_ = 0;
  std::string matrix_type_;
  int expected_max_ = 0;
};

namespace {

TEST_P(IvanovaPMaxMatrixFuncTests, FindMatrixMax) {
  ExecuteTest(GetParam());
}

class IvanovaPMaxMatrixSpecialTests : public ::testing::Test {
 protected:
  static void TestEmptyMatrix() {
    InType empty_matrix;
    IvanovaPMaxMatrixSEQ task(empty_matrix);
    EXPECT_FALSE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
  }

  static void TestJaggedMatrix() {
    InType jagged_matrix = {{1, 2}, {3}};
    IvanovaPMaxMatrixSEQ task(jagged_matrix);
    EXPECT_FALSE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
  }

  static void TestSingleElement() {
    InType single_element = {{42}};
    IvanovaPMaxMatrixSEQ task(single_element);
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    EXPECT_EQ(task.GetOutput(), 42);
  }

  static void TestKnownMaxMatrix() {
    InType matrix = {{1, 2, 1}, {2, 1, 2}, {1, 3, 1}};
    IvanovaPMaxMatrixSEQ task(matrix);
    EXPECT_TRUE(task.Validation());
    EXPECT_TRUE(task.PreProcessing());
    EXPECT_TRUE(task.Run());
    EXPECT_TRUE(task.PostProcessing());
    EXPECT_EQ(task.GetOutput(), 3);
  }
};

// Существующие тесты
TEST_F(IvanovaPMaxMatrixSpecialTests, EmptyMatrix) {
  TestEmptyMatrix();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, JaggedMatrix) {
  TestJaggedMatrix();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, SingleElement) {
  TestSingleElement();
}

TEST_F(IvanovaPMaxMatrixSpecialTests, KnownMaxMatrix) {
  TestKnownMaxMatrix();
}

const std::array<TestType, 6> kTestMatrices = {std::make_tuple(10, "small"),    std::make_tuple(100, "medium"),
                                               std::make_tuple(500, "large"),   std::make_tuple(1000, "xlarge"),
                                               std::make_tuple(50, "negative"), std::make_tuple(128, "rectangular")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<IvanovaPMaxMatrixMPI, InType>(kTestMatrices, PPC_SETTINGS_ivanova_p_max_matrix),
    ppc::util::AddFuncTask<IvanovaPMaxMatrixSEQ, InType>(kTestMatrices, PPC_SETTINGS_ivanova_p_max_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = IvanovaPMaxMatrixFuncTests::PrintFuncTestName<IvanovaPMaxMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixMaxTests, IvanovaPMaxMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ivanova_p_max_matrix
