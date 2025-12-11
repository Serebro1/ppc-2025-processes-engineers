#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstdint>
#include <string>
#include <tuple>

#include "kolotukhin_a_elem_vec_sum/common/include/common.hpp"
#include "kolotukhin_a_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "kolotukhin_a_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kolotukhin_a_elem_vec_sum {

class KolotukhinAElemVecSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    TestType tt = std::get<static_cast<std::int64_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::uint64_t size = std::get<0>(tt);
    input_data_ = size;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    TestType tt = std::get<static_cast<std::int64_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::int64_t test_answer = std::get<1>(tt);
    return output_data == test_answer;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(KolotukhinAElemVecSumFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}
const std::array<TestType, 5> kTestParam = {std::make_tuple(0, 0), std::make_tuple(1, 553), std::make_tuple(20, 79630),
                                            std::make_tuple(123, 537684), std::make_tuple(10000, 50015000)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KolotukhinAElemVecSumMPI, InType>(kTestParam, PPC_SETTINGS_kolotukhin_a_elem_vec_sum),
    ppc::util::AddFuncTask<KolotukhinAElemVecSumSEQ, InType>(kTestParam, PPC_SETTINGS_kolotukhin_a_elem_vec_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KolotukhinAElemVecSumFuncTests::PrintFuncTestName<KolotukhinAElemVecSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KolotukhinAElemVecSumFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kolotukhin_a_elem_vec_sum
