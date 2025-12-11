#include <gtest/gtest.h>
#include <mpi.h>  // Добавьте этот include

#include <algorithm>
#include <cstddef>
#include <vector>

#include "trofimov_n_max_val_matrix/common/include/common.hpp"
#include "trofimov_n_max_val_matrix/mpi/include/ops_mpi.hpp"
#include "trofimov_n_max_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace trofimov_n_max_val_matrix {

namespace {
constexpr int kDefaultMatrixSize = 5000;
constexpr int kRootRank = 0;
}  // namespace

class MaxValMatrixRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int k_matrix_size = kDefaultMatrixSize;

  InType input_data;
  OutType expected_output;

  void SetUp() override {
    input_data.clear();
    expected_output.clear();

    input_data.reserve(static_cast<std::size_t>(k_matrix_size));
    expected_output.reserve(static_cast<std::size_t>(k_matrix_size));

    for (int i = 0; i < k_matrix_size; ++i) {
      std::vector<int> row;
      row.reserve(static_cast<std::size_t>(k_matrix_size));

      for (int j = 0; j < k_matrix_size; ++j) {
        row.push_back((i * k_matrix_size) + j);
      }
      input_data.push_back(row);

      expected_output.push_back(*std::ranges::max_element(input_data.back()));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      int world_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

      if (world_rank != kRootRank) {
        return true;
      }
    }

    if (output_data.size() != expected_output.size()) {
      return false;
    }

    for (std::size_t i = 0; i < expected_output.size(); ++i) {
      if (output_data[i] != expected_output[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(MaxValMatrixRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TrofimovNMaxValMatrixMPI, TrofimovNMaxValMatrixSEQ>(
    PPC_SETTINGS_trofimov_n_max_val_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MaxValMatrixRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MaxValMatrixRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace trofimov_n_max_val_matrix
