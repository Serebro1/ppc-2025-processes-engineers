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
    std::string filename = SanitizeFileName(std::get<0>(test_param));
    std::string avg_str = FormatAverage(std::get<1>(test_param));
    return filename + "_" + avg_str;
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

  static std::string SanitizeFileName(const std::string &filename) {
    size_t lastindex = filename.find_last_of(".");
    std::string base_name = filename;
    if (lastindex != std::string::npos) {
        base_name = filename.substr(0, lastindex);
    }

    std::string safe_name = base_name;
    for (char &c : safe_name) {
      if (!std::isalnum(c) && c != '_') {
        c = '_';
      }
    }
    return safe_name;
  }

  static std::string FormatAverage(double value) {
    std::string str = RemoveTrailingZeros(value);
    if (value < 0) {
      str = "minus_" + str.substr(1, str.size());
    }

    for (char &c : str) {
      if (c == '.') {
        c = 'p';
      }
    }
    return "num_" + str;
  }

  static std::string RemoveTrailingZeros(double value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << value;
    std::string str = ss.str();
    if(str.find('.') != std::string::npos) {
        str = str.substr(0, str.find_last_not_of('0') + 1);
        if(str.find('.') == str.size() - 1) {
            str = str.substr(0, str.size() - 1);
        }
    }
    return str;
  }
};

namespace {

TEST_P(OtcheskovSElemVecAvgFuncTests, VectorAverageTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
  std::make_tuple("test_vec1.txt", 50.5),
  std::make_tuple("test_vec2.txt", 14.5),
  std::make_tuple("test_vec_one_elem.txt", 5.0),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSElemVecAvgMPI, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg),
                   ppc::util::AddFuncTask<OtcheskovSElemVecAvgSEQ, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = OtcheskovSElemVecAvgFuncTests::PrintFuncTestName<OtcheskovSElemVecAvgFuncTests>;

INSTANTIATE_TEST_SUITE_P(VectorAverageTests, OtcheskovSElemVecAvgFuncTests, kGtestValues, kFuncTestName);

TEST_F(OtcheskovSElemVecAvgFuncTests, EmptyVector) {
  SetUp();
  
}

}  // namespace

}  // namespace otcheskov_s_elem_vec_avg
