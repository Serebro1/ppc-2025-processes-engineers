# Введение
В задании 13 предлагается оптимизировать задачу поиска максимума в матрице, используя библиотеку MPI.

# Постановка задачи
Необходимо реализовать параллельный алгоритм поиска максимума в матрице, и сравнить его с последовательной версией.
Ограничения:
* Входные данные вектор вещественных чисел;
* Выходные данные - одно вещественное число, максимум матрицы.
# Описание алгоритма
Последовательная версия алгоритма уже реализована функцией std::ranges::max_element().
Её сложность, согласно сайту en.cppreference.com, равна N = std::distance(first, last). А в нашем случае длине вектора.
# Схема распараллеливания
В параллельной программе процесс с номером 0 распределяет данные входного вектора между рабочими процессами поровну(если количество элементов вектора не делиться нацело на количество процессов, то добавляются фиктивные элементы). Далее в каждом процессе выполняется последовательная версия алгоритма - вызов функции std::ranges::max_element(), так мы находим локальные максимумы. Глобальный максимум находится с помощью функции MPI_Allreduce с операцией MPI_MAX.
Полный код реализации в приложении.
# Экспериментальные результаты
| Режим выполнения | Количество процессов | Время выполнения (сек) | Ускорение | Эффективность |
|------------------|---------------------|------------------------|-----------|---------------|
| **Последовательный** | | | | |
| pipeline | 1 | 0.0014113600 | 1.00x | — |
| task_run | 1 | 0.0012853000 | 1.00x | — |
| **MPI (2 процесса)** | | | | |
| pipeline | 2 | 0.0113958000 | 0.12x | 6% |
| task_run | 2 | 0.0107858200 | 0.12x | 6% |
| **MPI (4 процесса)** | | | | |
| pipeline | 4 | 0.0118968200 | 0.12x | 3% |
| task_run | 4 | 0.0119689000 | 0.11x | 3% |
| **MPI (6 процессов)** | | | | |
| pipeline | 6 | 0.0166549200 | 0.08x | 1% |
| task_run | 6 | 0.0164631400 | 0.08x | 1% |
## Тестовые данные

Элементами матрицы для тестов на производительность являются числа от 0 до 4000. Элементов в матрице 4 миллиона. 
Полный код задания матрицы в приложении.
## Окружение
| Параметр       | Значение                                          |
| -------------- | ------------------------------------------------- |
| **OS**         | Windows 11 Pro 25H2                               |
| **CPU**        | AMD Ryzen 5 5600X (6 cores, 12 threads, 3.70 GHz) |
| **RAM**        | 16 GB DDR4 3400 МГц                               |
| **Компилятор** | MSVC 14.43.34808                                  |
| **Версия MPI** | Microsoft MPI 10.1.12498.52                       |


# Результаты

Хотя оба алгоритма успешно проходят функциональные тесты, последовательная версия значительно производительнее. Связано это, наиболее вероятно, с неэффективной параллельной реализацией: для такого количества данных накладные расходы на общение между процессами нерационально высоки.

# Источники
1. https://en.cppreference.com/w/cpp/algorithm/ranges/max_element - веб-страница с информацией про std::ranges::max_element().
## Код параллельной версии

```
bool GolovanovDMatrixMaxElemMPI::RunImpl() {
  int rank = 0;
  int processes = 0;
  int n = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &processes);
  std::vector<double> elems;
  double answer = -1234;
  if (rank == 0) {
    auto columns = std::get<0>(GetInput());
    auto strokes = std::get<1>(GetInput());
    elems = std::get<2>(GetInput());
    auto count = columns * strokes;

    n = count / processes;
    if (count % processes != 0) {
      for (int i = 0; i < processes - (count % processes); i++) {
        elems.push_back(elems[0]);
      }
      n++;
    }
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<double> work_vector(n);
  MPI_Scatter(elems.data(), n, MPI_DOUBLE, work_vector.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double max = *std::ranges::max_element(work_vector);
  MPI_Allreduce(&max, &answer, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  GetOutput() = answer;
  return true;
}
```
## Код создания матрицы для тестов производительности
```
void SetUp() override {
    std::vector<double> tmp_vector(0);
    int n = 2000;
    int m = 2000;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        double value = i + j;
        tmp_vector.push_back(value);
      }
    }
    tmp_vector[10] = maximum_;
    input_data_ = std::tuple<int, int, std::vector<double>>(n, m, tmp_vector);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == maximum_;
  }
```