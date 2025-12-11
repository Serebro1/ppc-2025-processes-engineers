#include "kotelnikova_a_num_sent_in_line/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>

#include "kotelnikova_a_num_sent_in_line/common/include/common.hpp"

namespace kotelnikova_a_num_sent_in_line {

KotelnikovaANumSentInLineMPI::KotelnikovaANumSentInLineMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = static_cast<std::size_t>(0);
}

bool KotelnikovaANumSentInLineMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool KotelnikovaANumSentInLineMPI::PreProcessingImpl() {
  return true;
}

bool KotelnikovaANumSentInLineMPI::RunImpl() {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int total_length = 0;
  if (world_rank == 0) {
    total_length = static_cast<int>(GetInput().length());
  }
  MPI_Bcast(&total_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::string text;
  if (world_rank == 0) {
    text = GetInput();
  } else {
    text.resize(static_cast<std::size_t>(total_length));
  }
  MPI_Bcast(text.data(), total_length, MPI_CHAR, 0, MPI_COMM_WORLD);

  int chunk_size = total_length / world_size;
  int remainder = total_length % world_size;
  int start = (world_rank * chunk_size) + std::min(world_rank, remainder);
  int end = start + chunk_size + (world_rank < remainder ? 1 : 0);
  end = std::min(end, total_length);

  int local_count = CountLocalSentences(text, start, end, total_length);

  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<std::size_t>(global_count);
  return true;
}

int KotelnikovaANumSentInLineMPI::CountLocalSentences(const std::string &text, int start, int end, int total_length) {
  int local_count = 0;
  bool local_in_sentence = false;

  local_in_sentence = BorderControl(text, start);

  if (start < end) {
    ProcessingPart(text, start, end, local_count, local_in_sentence);
  }

  if (local_in_sentence && end == total_length) {
    local_count++;
  }

  return local_count;
}

bool KotelnikovaANumSentInLineMPI::BorderControl(const std::string &text, int start) {
  int pos = start - 1;
  while (pos >= 0) {
    char c = text[static_cast<std::size_t>(pos)];
    if (c == '.' || c == '!' || c == '?') {
      return false;
    }
    if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
      return true;
    }
    pos--;
  }
  return false;
}

void KotelnikovaANumSentInLineMPI::ProcessingPart(const std::string &text, int start, int end, int &local_count,
                                                  bool &local_in_sentence) {
  for (int i = start; i < end; ++i) {
    char c = text[static_cast<std::size_t>(i)];

    if (c == '.' || c == '!' || c == '?') {
      if (local_in_sentence) {
        local_count++;
        local_in_sentence = false;
      }
    }

    if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
      local_in_sentence = true;
    }
  }
}

bool KotelnikovaANumSentInLineMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kotelnikova_a_num_sent_in_line
