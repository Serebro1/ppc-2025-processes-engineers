#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string filename = std::get<0>(test_param);
    int expect_avg = std::get<1>(test_param);

    size_t lastindex = filename.find_last_of(".");
    if (lastindex != std::string::npos) {
        filename = filename.substr(0, lastindex);
    }

    std::string safe_filename = filename;
    for (char &c : safe_filename) {
      if (!std::isalnum(c) && c != '_') {
        c = '_';
      }
    }

    std::string avg_str;
    if (expect_avg < 0) {
      avg_str = "minus_" + std::to_string(-expect_avg);
    } else {
      avg_str = std::to_string(expect_avg);
    }
    
    return safe_filename + "_" + avg_str;
  }
 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string filename = std::get<0>(params);
    expected_avg_ = std::get<1>(params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_otcheskov_s_elem_vec_avg, filename);
    std::ifstream file(abs_path);
    
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }

    int num;
    while (file >> num) {
      input_data_.push_back(num);
    }
    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_avg_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_avg_;
};

namespace {

TEST_P(OtcheskovSElemVecAvgFuncTests, VectorAverageTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
  std::make_tuple("test_vec1.txt", 50),
  std::make_tuple("test_vec2.txt", 14),
  std::make_tuple("test_vec_one_elem.txt", 5),
  std::make_tuple("test_vec_one_minus_elem.txt", -5)
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSElemVecAvgMPI, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg),
                   ppc::util::AddFuncTask<OtcheskovSElemVecAvgSEQ, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = OtcheskovSElemVecAvgFuncTests::PrintFuncTestName<OtcheskovSElemVecAvgFuncTests>;

INSTANTIATE_TEST_SUITE_P(VectorAverageTests, OtcheskovSElemVecAvgFuncTests, kGtestValues, kPerfTestName);


}  // namespace

}  // namespace otcheskov_s_elem_vec_avg
