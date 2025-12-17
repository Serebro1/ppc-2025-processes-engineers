#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <iostream>

#include "otcheskov_s_linear_topology/common/include/common.hpp"
#include "otcheskov_s_linear_topology/mpi/include/ops_mpi.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_linear_topology {

class OtcheskovSLinearTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 128000000;
  InType input_msg_;

  void SetUp() override {
    input_msg_.first = {.delivered = 0, .src = 0, .dest = 1, .data_size = 0};
    input_msg_.second.resize(kDataSize);
    for (int i = 0; i < kDataSize; ++i) {
      input_msg_.second[static_cast<std::size_t>(i)] = i;
    }
  }

  bool CheckTestOutputData(OutType &output_msg) final {
    bool is_valid = false;
    const auto &[in_header, in_data] = input_msg_;
    const auto &[out_header, out_data] = output_msg;
    if (!ppc::util::IsUnderMpirun()) {
      is_valid = out_header.delivered != 0;
    } else {
      int proc_rank{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

      if (proc_rank == in_header.src || proc_rank == in_header.dest) {
        is_valid = (in_data == out_data) && out_header.delivered != 0;
      } else {
        is_valid = true;
      }
    }
    return is_valid;
  }

  InType GetTestInputData() final {
    return input_msg_;
  }
};

TEST_P(OtcheskovSLinearTopologyPerfTests, LinearTopologyPerfTests) {
  if (!ppc::util::IsUnderMpirun()) {
    ExecuteTest(GetParam());
  } else {
    int proc_nums{};
    MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);
    if (proc_nums < 2) {
      std::cerr << "Tests should run on 2 or more processes\n";
    } else {
      ExecuteTest(GetParam());
    }
  }
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OtcheskovSLinearTopologyMPI>(PPC_SETTINGS_otcheskov_s_linear_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OtcheskovSLinearTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(LinearTopologyPerfTests, OtcheskovSLinearTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace otcheskov_s_linear_topology
