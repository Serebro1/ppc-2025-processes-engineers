#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

#include "kotelnikova_a_num_sent_in_line/common/include/common.hpp"
#include "kotelnikova_a_num_sent_in_line/mpi/include/ops_mpi.hpp"
#include "kotelnikova_a_num_sent_in_line/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kotelnikova_a_num_sent_in_line {

class KotelnikovaARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data;
  std::size_t expected_count{0};

  void SetUp() override {
    input_data = LoadTestData();
    expected_count = 1;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_count;
  }

  InType GetTestInputData() final {
    return input_data;
  }

  static std::string LoadTestData() {
    std::string path = "tasks/kotelnikova_a_num_sent_in_line/data/test_7.txt";
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }
};

TEST_P(KotelnikovaARunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KotelnikovaANumSentInLineMPI, KotelnikovaANumSentInLineSEQ>(
        PPC_SETTINGS_kotelnikova_a_num_sent_in_line);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KotelnikovaARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KotelnikovaARunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kotelnikova_a_num_sent_in_line
