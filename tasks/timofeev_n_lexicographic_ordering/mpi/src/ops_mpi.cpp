#include "timofeev_n_lexicographic_ordering/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>

#include "timofeev_n_lexicographic_ordering/common/include/common.hpp"

namespace timofeev_n_lexicographic_ordering {

TimofeevNLexicographicOrderingMPI::TimofeevNLexicographicOrderingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::pair<int, int>(1, 1);
}

bool TimofeevNLexicographicOrderingMPI::ValidationImpl() {
  return true;
}

bool TimofeevNLexicographicOrderingMPI::PreProcessingImpl() {
  return true;
}

bool TimofeevNLexicographicOrderingMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  GetOutput() = std::pair<int, int>(1, 1);

  if (size <= 1) {
    auto input = GetInput();
    for (size_t i = 0; !input.first.empty() && i < input.first.length() - 1; i++) {
      GetOutput().first &= static_cast<int>(input.first[i] <= input.first[i + 1]);
    }
    for (size_t i = 0; !input.second.empty() && i < input.second.length() - 1; i++) {
      GetOutput().second &= static_cast<int>(input.second[i] <= input.second[i + 1]);
    }
    return true;
  }

  if (rank == 0) {
    auto input = GetInput();
    size_t second_length = input.second.length();
    MPI_Send(&second_length, 1, MPI_UNSIGNED_LONG, 1, 0, MPI_COMM_WORLD);
    MPI_Send(input.second.c_str(), static_cast<int>(second_length), MPI_CHAR, 1, 1, MPI_COMM_WORLD);
    // only true if comparison is true on every step
    for (size_t i = 0; !input.first.empty() && i < input.first.length() - 1; i++) {
      GetOutput().first &= static_cast<int>(input.first[i] <= input.first[i + 1]);
    }
  } else if (rank == 1) {
    size_t llength = 0;
    // если упадёт на каких-то мудрёных тестах - может быть, дело в size_t
    MPI_Recv(&llength, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    char *inpput = new char[(llength + 1) * sizeof(char)];
    inpput[llength] = '\0';
    MPI_Recv(inpput, static_cast<int>(llength), MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (size_t i = 0; llength != 0 && i < llength - 1; i++) {
      GetOutput().second &= static_cast<int>(inpput[i] <= inpput[i + 1]);
    }
    delete[] inpput;
  }

  MPI_Bcast(&GetOutput().first, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&GetOutput().second, 1, MPI_INT, 1, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool TimofeevNLexicographicOrderingMPI::PostProcessingImpl() {
  // idk what else to put there
  return true;
}

}  // namespace timofeev_n_lexicographic_ordering
