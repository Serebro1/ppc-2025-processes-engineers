#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zorin_d_avg_vec/common/include/common.hpp"
#include "zorin_d_avg_vec/mpi/include/ops_mpi.hpp"
#include "zorin_d_avg_vec/seq/include/ops_seq.hpp"

namespace zorin_d_avg_vec {

class ZorinDAvgVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &info) {
    const TestType &p = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);
    std::string name = std::get<2>(p);

    for (char &c : name) {
      if (std::isalnum(c) == 0) {
        c = '_';
      }
    }
    const std::string &task_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);

    return task_name + "_" + name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const std::string &filename = std::get<2>(params);
    expected_ = std::get<1>(params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_zorin_d_avg_vec, filename);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("File not found: " + abs_path);
    }

    input_.clear();
    double value = 0.0;
    while (file >> value) {
      input_.push_back(static_cast<int>(value));
    }
  }

  bool CheckTestOutputData(OutType &output) final {
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (initialized == 0) {
      return std::fabs(output - expected_) < 1e-6;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double ref_value = output;
    MPI_Bcast(&ref_value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return std::fabs(ref_value - expected_) < 1e-6;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
  OutType expected_{};
};

namespace {

TEST_P(ZorinDAvgVecFuncTests, FunctionalAverageCheck) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParams = {{
    TestType({}, 55.0, "avg_vec_data.txt"),
    TestType({}, 5.0, "avg_vec_equal.txt"),
    TestType({}, 0.0, "avg_vec_negative.txt"),
    TestType({}, 42.0, "avg_vec_single.txt"),
    TestType({}, 0.0, "avg_vec_empty.txt"),
}};

const auto kTasks =
    std::tuple_cat(ppc::util::AddFuncTask<ZorinDAvgVecMPI, InType>(kTestParams, PPC_SETTINGS_zorin_d_avg_vec),
                   ppc::util::AddFuncTask<ZorinDAvgVecSEQ, InType>(kTestParams, PPC_SETTINGS_zorin_d_avg_vec));

INSTANTIATE_TEST_SUITE_P(AvgVecFunc, ZorinDAvgVecFuncTests, ppc::util::ExpandToValues(kTasks),
                         ZorinDAvgVecFuncTests::PrintTestParam);

}  // namespace

}  // namespace zorin_d_avg_vec
