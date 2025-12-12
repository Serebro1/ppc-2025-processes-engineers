#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "golovanov_d_matrix_max_elem//common/include/common.hpp"
#include "golovanov_d_matrix_max_elem//mpi/include/ops_mpi.hpp"
#include "golovanov_d_matrix_max_elem//seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace golovanov_d_matrix_max_elem {

class GolovanovDMatrixMaxElemFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string s1 = std::to_string(std::get<0>(test_param)) + "_";
    std::string s2 = std::to_string(std::get<1>(test_param)) + "_";
    std::string s3;
    if (std::get<2>(test_param) < 0) {
      s3 = "minus" + std::to_string(static_cast<int>(-std::get<2>(test_param)));
    } else {
      s3 = std::to_string(static_cast<int>(std::get<2>(test_param)));
    }
    return s1 + s2 + s3;
  }

 protected:
  double maximum{};
  int max_pos{};
  std::mt19937 gen{std::random_device{}()};

  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::vector<double> tmp_vector(0);
    int n = std::get<0>(params);
    int m = std::get<1>(params);
    maximum = std::get<2>(params);

    std::uniform_real_distribution<double> real_dist(-1000000, maximum);
    std::uniform_int_distribution<int> int_dist(0, (n * m) - 1);

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        tmp_vector.push_back(real_dist(gen));
      }
    }
    max_pos = int_dist(gen);
    tmp_vector[max_pos] = maximum;
    input_data_ = std::tuple<int, int, std::vector<double>>(n, m, tmp_vector);
  }
  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == maximum;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(GolovanovDMatrixMaxElemFuncTest, TestTest1) {
  ExecuteTest(GetParam());
}
const std::array<TestType, 4> kTestParam = {TestType(5, 5, 10.0), TestType(5, 5, -10.0), TestType(5, 5, 0),
                                            TestType(1, 1, 100)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GolovanovDMatrixMaxElemMPI, InType>(kTestParam, PPC_SETTINGS_golovanov_d_matrix_max_elem),
    ppc::util::AddFuncTask<GolovanovDMatrixMaxElemSEQ, InType>(kTestParam, PPC_SETTINGS_golovanov_d_matrix_max_elem));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GolovanovDMatrixMaxElemFuncTest::PrintFuncTestName<GolovanovDMatrixMaxElemFuncTest>;

INSTANTIATE_TEST_SUITE_P(MatrixMaxElemFunTests, GolovanovDMatrixMaxElemFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace golovanov_d_matrix_max_elem
