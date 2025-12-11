#include <gtest/gtest.h>

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "gaivoronskiy_m_average_vector_sum/common/include/common.hpp"
#include "gaivoronskiy_m_average_vector_sum/mpi/include/ops_mpi.hpp"
#include "gaivoronskiy_m_average_vector_sum/seq/include/ops_seq.hpp"
#include "performance/include/performance.hpp"
#include "task/include/task.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace gaivoronskiy_m_average_vector_sum {

namespace {

constexpr std::array<std::size_t, 4> kPerfSizes = {1'000'000, 5'000'000, 20'000'000, 100'000'000};
constexpr std::string_view kPerfBaseFile = "perf_vec_base.txt";

std::vector<double> LoadVectorFromFile(const std::string &file_name) {
  std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_gaivoronskiy_m_average_vector_sum, file_name);
  std::ifstream file(abs_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + abs_path);
  }
  std::vector<double> data;
  double value = 0.0;
  while (file >> value) {
    data.push_back(value);
  }
  return data;
}

}  // namespace

class GaivoronskiyRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const auto &test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
    const std::size_t data_size = ResolveInputSizeFromTestParam(test_name);
    const auto base_pattern = LoadVectorFromFile(std::string(kPerfBaseFile));
    if (base_pattern.empty()) {
      throw std::runtime_error("Performance base vector file is empty");
    }
    input_data_.clear();
    input_data_.reserve(data_size);
    while (input_data_.size() < data_size) {
      const std::size_t remaining = data_size - input_data_.size();
      if (remaining >= base_pattern.size()) {
        input_data_.insert(input_data_.end(), base_pattern.begin(), base_pattern.end());
      } else {
        input_data_.insert(input_data_.end(), base_pattern.begin(),
                           base_pattern.begin() + static_cast<std::ptrdiff_t>(remaining));
      }
    }
    expected_average_ = CalculateAverage(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double k_eps = 1e-9;
    return std::fabs(output_data - expected_average_) <= k_eps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  static std::size_t ResolveInputSizeFromTestParam(const std::string &test_name) {
    constexpr std::string_view kSizeTag = "_size";
    const auto pos = test_name.rfind(kSizeTag);
    if (pos == std::string::npos) {
      return kPerfSizes.front();
    }
    const auto suffix = std::string_view(test_name).substr(pos + kSizeTag.size());
    std::size_t parsed_value = 0;
    const auto result = std::from_chars(suffix.data(), suffix.data() + suffix.size(), parsed_value);
    if (result.ec != std::errc() || parsed_value == 0) {
      return kPerfSizes.front();
    }
    return parsed_value;
  }

  static double CalculateAverage(const InType &values) {
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
  }

  InType input_data_;
  OutType expected_average_ = 0.0;
};

TEST_P(GaivoronskiyRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

template <typename TaskType>
auto MakePerfTaskTuplesForSize(std::size_t size, const std::string &settings_path) {
  const auto base_name = std::string(ppc::util::GetNamespace<TaskType>()) + "_" +
                         ppc::task::GetStringTaskType(TaskType::GetStaticTypeOfTask(), settings_path);
  const auto size_suffix = "_size" + std::to_string(size);
  const auto task_name = base_name + size_suffix;
  return std::make_tuple(std::make_tuple(ppc::task::TaskGetter<TaskType, InType>, task_name,
                                         ppc::performance::PerfResults::TypeOfRunning::kPipeline),
                         std::make_tuple(ppc::task::TaskGetter<TaskType, InType>, task_name,
                                         ppc::performance::PerfResults::TypeOfRunning::kTaskRun));
}

template <typename TaskType, std::size_t N, std::size_t... I>
auto MakePerfTasksForAllSizesImpl(const std::array<std::size_t, N> &sizes, const std::string &settings_path,
                                  std::index_sequence<I...> /*unused*/) {
  return std::tuple_cat(MakePerfTaskTuplesForSize<TaskType>(sizes[I], settings_path)...);
}

template <typename TaskType, std::size_t N>
auto MakePerfTasksForAllSizes(const std::array<std::size_t, N> &sizes, const std::string &settings_path) {
  return MakePerfTasksForAllSizesImpl<TaskType>(sizes, settings_path, std::make_index_sequence<N>{});
}

}  // namespace

const auto kAllPerfTasks = std::tuple_cat(
    MakePerfTasksForAllSizes<GaivoronskiyMAverageVecSumMPI>(kPerfSizes, PPC_SETTINGS_gaivoronskiy_m_average_vector_sum),
    MakePerfTasksForAllSizes<GaivoronskiyMAverageVecSumSEQ>(kPerfSizes,
                                                            PPC_SETTINGS_gaivoronskiy_m_average_vector_sum));

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GaivoronskiyRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GaivoronskiyRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace gaivoronskiy_m_average_vector_sum
