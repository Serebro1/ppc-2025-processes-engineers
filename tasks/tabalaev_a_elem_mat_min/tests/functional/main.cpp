#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "tabalaev_a_elem_mat_min/common/include/common.hpp"
#include "tabalaev_a_elem_mat_min/mpi/include/ops_mpi.hpp"
#include "tabalaev_a_elem_mat_min/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tabalaev_a_elem_mat_min {

class TabalaevAElemMatMinFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int minik = std::get<2>(test_param);
    std::string str_minik = (minik < 0) ? "minus" + std::to_string(-minik) : std::to_string(minik);
    return std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param)) + "_min_" +
           str_minik + "_" + std::get<3>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int rows = std::get<0>(params);
    int columns = std::get<1>(params);
    int minik = std::get<2>(params);

    std::vector<int> matrix(static_cast<size_t>(rows) * static_cast<size_t>(columns));

    std::uniform_int_distribution<int> dist(minik, 250);

    for (int &elem : matrix) {
      elem = dist(gen_);
    }

    matrix[(rows * columns) / 2] = minik;

    input_data_ = std::make_tuple(static_cast<size_t>(rows), static_cast<size_t>(columns), matrix);
    expected_minik_ = minik;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_minik_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_minik_ = 0;
  std::mt19937 gen_{std::random_device{}()};
};

namespace {

TEST_P(TabalaevAElemMatMinFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(3, 3, -15, "Small_matrix"),
                                            std::make_tuple(5, 5, 32, "Medium_matrix"),
                                            std::make_tuple(10, 10, -41, "Large_matrix"),
                                            std::make_tuple(10, 5, 4, "Different_size_matrix_rows"),
                                            std::make_tuple(6, 11, -1, "Different_size_matrix_columns"),
                                            std::make_tuple(1, 1, -10, "Only_1_elem")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TabalaevAElemMatMinMPI, InType>(kTestParam, PPC_SETTINGS_tabalaev_a_elem_mat_min),
    ppc::util::AddFuncTask<TabalaevAElemMatMinSEQ, InType>(kTestParam, PPC_SETTINGS_tabalaev_a_elem_mat_min));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TabalaevAElemMatMinFuncTests::PrintFuncTestName<TabalaevAElemMatMinFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TabalaevAElemMatMinFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tabalaev_a_elem_mat_min
