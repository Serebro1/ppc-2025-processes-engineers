#include "korolev_k_string_word_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>

#include "korolev_k_string_word_count/common/include/common.hpp"

namespace korolev_k_string_word_count {

namespace {

int CountWordsChunk(const std::string &s, std::size_t begin, std::size_t end, bool prev_is_space) {
  if (begin >= end) {
    return 0;
  }
  int count = 0;
  bool in_word = false;

  auto first = static_cast<unsigned char>(s[begin]);
  bool first_is_space = std::isspace(first) != 0;
  if (!first_is_space && prev_is_space) {
    ++count;
  }
  in_word = !first_is_space;

  for (std::size_t i = begin + 1; i < end; ++i) {
    auto c = static_cast<unsigned char>(s[i]);
    bool is_space = std::isspace(c) != 0;
    if (!is_space) {
      if (!in_word) {
        ++count;
        in_word = true;
      }
    } else {
      in_word = false;
    }
  }
  return count;
}

}  // namespace

KorolevKStringWordCountMPI::KorolevKStringWordCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KorolevKStringWordCountMPI::ValidationImpl() {
  // Any string is valid.
  return GetOutput() == 0;
}

bool KorolevKStringWordCountMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KorolevKStringWordCountMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string s;
  uint64_t n = 0;

  if (rank == 0) {
    s = GetInput();
    n = static_cast<uint64_t>(s.size());
  }

  MPI_Bcast(&n, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  if (n == 0) {
    GetOutput() = 0;
    return true;
  }

  const auto n_size = static_cast<std::size_t>(n);
  const auto size_z = static_cast<std::size_t>(size);
  const std::size_t base = n_size / size_z;
  const std::size_t rem = n_size % size_z;

  std::string local_segment;
  char prev_char = ' ';

  if (rank == 0) {
    for (int dest = 1; dest < size; ++dest) {
      auto dest_z = static_cast<std::size_t>(dest);
      std::size_t begin_i = (dest_z * base) + std::min(dest_z, rem);
      std::size_t end_i = begin_i + base + (dest_z < rem ? std::size_t{1} : std::size_t{0});
      int segment_len = static_cast<int>(end_i - begin_i);

      MPI_Send(&segment_len, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

      char prev = (begin_i > 0) ? s[begin_i - 1] : ' ';
      MPI_Send(&prev, 1, MPI_CHAR, dest, 1, MPI_COMM_WORLD);

      MPI_Send(s.data() + begin_i, segment_len, MPI_CHAR, dest, 2, MPI_COMM_WORLD);
    }

    std::size_t begin_0 = 0;
    std::size_t end_0 = base + (rem > 0 ? std::size_t{1} : std::size_t{0});
    local_segment = s.substr(begin_0, end_0 - begin_0);
    prev_char = ' ';
  } else {
    int segment_len = 0;
    MPI_Recv(&segment_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&prev_char, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_segment.resize(static_cast<std::size_t>(segment_len));
    MPI_Recv(local_segment.data(), segment_len, MPI_CHAR, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  bool prev_is_space = std::isspace(static_cast<unsigned char>(prev_char)) != 0;

  int local_count = CountWordsChunk(local_segment, 0, local_segment.size(), prev_is_space);

  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_count;

  return true;
}

bool KorolevKStringWordCountMPI::PostProcessingImpl() {
  // Nothing to post-process.
  return true;
}

}  // namespace korolev_k_string_word_count
