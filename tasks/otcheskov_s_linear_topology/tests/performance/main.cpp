#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "otcheskov_s_linear_topology/common/include/common.hpp"
#include "otcheskov_s_linear_topology/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace otcheskov_s_linear_topology {

class OtcheskovSLinearTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 128000000;
  InType input_data_;

  void SetUp() override {
    input_data_ = {0, 1, {}, false};
    input_data_.data.resize(kDataSize);
    for (int i = 0; i < kDataSize; ++i) {
      input_data_.data[static_cast<std::size_t>(i)] = i;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool is_valid = false;
    if (!ppc::util::IsUnderMpirun()) {
      is_valid = output_data.delivered;
    } else {
      int proc_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
      if (proc_rank == input_data_.src || proc_rank == input_data_.dest) {
        is_valid = (input_data_.data == output_data.data) && output_data.delivered;
      } else {
        is_valid = true;
      }
    }
    return is_valid;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(OtcheskovSLinearTopologyPerfTests, LinearTopologyPerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OtcheskovSLinearTopologyMPI>(PPC_SETTINGS_otcheskov_s_linear_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OtcheskovSLinearTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(LinearTopologyPerfTests, OtcheskovSLinearTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace otcheskov_s_linear_topology
