#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kopilov_d_sum_val_col_mat/common/include/common.hpp"
#include "kopilov_d_sum_val_col_mat/mpi/include/ops_mpi.hpp"
#include "kopilov_d_sum_val_col_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kopilov_d_sum_val_col_mat {

class KopilovDSumValColMatTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(std::get<0>(param)) + "_" + std::get<1>(param);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    matrix_size_ = std::get<0>(params);

    input_.rows = matrix_size_;
    input_.cols = matrix_size_;
    input_.data.resize(static_cast<std::size_t>(matrix_size_) * static_cast<std::size_t>(matrix_size_));

    for (int row = 0; row < matrix_size_; ++row) {
      for (int col = 0; col < matrix_size_; ++col) {
        input_.data[(static_cast<std::size_t>(row) * static_cast<std::size_t>(matrix_size_)) +
                    static_cast<std::size_t>(col)] = static_cast<double>(row + col);
      }
    }

    expected_.col_sum.resize(static_cast<std::size_t>(matrix_size_));
    for (int col = 0; col < matrix_size_; ++col) {
      double sum = 0.0;
      for (int row = 0; row < matrix_size_; ++row) {
        sum += static_cast<double>(row + col);
      }
      expected_.col_sum[static_cast<std::size_t>(col)] = sum;
    }
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    if (output_data.col_sum.size() != expected_.col_sum.size()) {
      return false;
    }
    for (std::size_t i = 0; i < expected_.col_sum.size(); ++i) {
      if (std::abs(output_data.col_sum[i] - expected_.col_sum[i]) > 1e-9) {
        return false;
      }
    }
    return true;
  }

 private:
  int matrix_size_ = 0;
  InType input_;
  OutType expected_;
};

namespace {

TEST_P(KopilovDSumValColMatTests, ColumnSumTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KopilovDSumValColMatMPI, InType>(kTestParam, PPC_SETTINGS_kopilov_d_sum_val_col_mat),
    ppc::util::AddFuncTask<KopilovDSumValColMatSEQ, InType>(kTestParam, PPC_SETTINGS_kopilov_d_sum_val_col_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = KopilovDSumValColMatTests::PrintFuncTestName<KopilovDSumValColMatTests>;

INSTANTIATE_TEST_SUITE_P(ColumnSumTests, KopilovDSumValColMatTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kopilov_d_sum_val_col_mat
