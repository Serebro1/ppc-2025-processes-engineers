#include "makoveeva_s_number_of_sentence/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "makoveeva_s_number_of_sentence/common/include/common.hpp"

namespace makoveeva_s_number_of_sentence {

SentencesCounterMPI::SentencesCounterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SentencesCounterMPI::ValidationImpl() {
  return true;
}

bool SentencesCounterMPI::PreProcessingImpl() {
  return true;
}

bool SentencesCounterMPI::IsSentenceEnding(char character) {
  return character == '.' || character == '!' || character == '?';
}

int SentencesCounterMPI::SkipRepeatedPunctuation(const std::string &text, int current_position) {
  auto position = static_cast<std::size_t>(current_position);
  while (position < text.length() && IsSentenceEnding(text[position])) {
    position++;
  }
  return static_cast<int>(position);
}

int SentencesCounterMPI::ProcessTextSegment(const std::string &text_segment, char previous_char) {
  int sentence_count = 0;
  std::size_t index = 0;
  const auto segment_length = text_segment.length();

  if (segment_length > 0) {
    char first_char = text_segment[0];

    if (IsSentenceEnding(previous_char) && IsSentenceEnding(first_char)) {
      index = static_cast<std::size_t>(SkipRepeatedPunctuation(text_segment, 0));
    }
  }

  while (index < segment_length) {
    char current_char = text_segment[index];

    if (IsSentenceEnding(current_char)) {
      sentence_count++;
      index = static_cast<std::size_t>(SkipRepeatedPunctuation(text_segment, static_cast<int>(index + 1)));
    } else {
      index++;
    }
  }

  return sentence_count;
}

bool SentencesCounterMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string full_text;
  int text_length = 0;

  if (rank == 0) {
    full_text = GetInput();
    text_length = static_cast<int>(full_text.length());
  }

  MPI_Bcast(&text_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (text_length == 0) {
    GetOutput() = 0;
    return true;
  }

  int base_chunk_size = text_length / size;
  int remaining_chars = text_length % size;

  std::vector<int> chunk_sizes(static_cast<std::size_t>(size));
  std::vector<int> displacements(static_cast<std::size_t>(size));

  int current_displacement = 0;
  for (int i = 0; i < size; ++i) {
    chunk_sizes[static_cast<std::size_t>(i)] = base_chunk_size + (i < remaining_chars ? 1 : 0);
    displacements[static_cast<std::size_t>(i)] = current_displacement;
    current_displacement += chunk_sizes[static_cast<std::size_t>(i)];
  }

  int local_chunk_size = chunk_sizes[static_cast<std::size_t>(rank)];
  std::string local_chunk(static_cast<std::size_t>(local_chunk_size), '\0');

  MPI_Scatterv(rank == 0 ? full_text.data() : nullptr, chunk_sizes.data(), displacements.data(), MPI_CHAR,
               local_chunk.data(), local_chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  std::vector<char> boundary_chars(static_cast<std::size_t>(size));
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      int chunk_start = displacements[static_cast<std::size_t>(i)];
      boundary_chars[static_cast<std::size_t>(i)] =
          (chunk_start > 0) ? full_text[static_cast<std::size_t>(chunk_start - 1)] : '\0';
    }
  }

  char previous_char = '\0';
  MPI_Scatter(boundary_chars.data(), 1, MPI_CHAR, &previous_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_sentence_count = ProcessTextSegment(local_chunk, previous_char);

  int total_sentences = 0;
  MPI_Reduce(&local_sentence_count, &total_sentences, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&total_sentences, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = total_sentences;
  return true;
}

bool SentencesCounterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace makoveeva_s_number_of_sentence
