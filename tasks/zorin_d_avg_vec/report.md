# Вычисление среднего значения элементов вектора

- Студент: Зорин Данила Артёмович, группа 3823Б1ПР2
- Технология: SEQ | MPI
- Вариант: 2

## 1. Вступление
Вычисление среднего значения элементов вектора является базовой операцией, широко применяемой в задачах анализа данных, статистической обработки, и т.д.. При небольших размерах входных данных вычисление среднего выполняется быстро и не требует оптимизации. Однако при обработке массивов больших размеров, например, в высокопроизводительных вычислениях или при работе с потоками данных, время выполнения существенно увеличивается.

Последовательный (SEQ) алгоритм обладает линейной временной сложностью (O(N)) и использует один поток выполнения. Для повышения производительности и сокращения времени вычислений применяются параллельные методы. Использование технологии MPI  позволяет распараллелить вычисления на кластерных системах и многопроцессорных архитектурах, распределив данные между несколькими процессами и уменьшив время обработки.

Цель данной работы - реализовать и сравнить последовательный и параллельный подходы к вычислению среднего значения элементов вектора, исследовать прирост производительности при увеличении количества процессов и оценить эффективность масштабируемости MPI-реализации.

## 2. Постановка задачи
У нас есть вектор из `n` элементов, где все элементы целочисленные (`int`). Необходимо вычислить среднее значение элементов вектора

### Входные данные
Входные файлы подаются из текстовых файлов. Передаются элементы вектора через пробел. 

Например, `10 20 30 40 50 60 70 80 90 100`

### Выходные данные
Результатом работы будет вещественное число (`double`) - означающие среднее значение элементов вектора

### Ограничения
Вектор может быть только из целочисленных элементов.

## 3. Последовательная версия (SEQ)
Последовательный алгоритм вычисления среднего значения элементов вектора реализован в классе `ZorinDAvgVecSEQ` и состоит из следующих этапов:

1. `ValidationImpl`
* Проверка входных данных
2. `PreProcessingImpl`
* Подготовительные действия перед выполнением основной логики
3. `RunImpl`
* Основной этап вычисления среднего значения
* Получение входных данных
* Проверка пустого вектора. Если вектор пустой, то `AVG` = `0.0` и завершение вычисление среднего значения
* Вычисление суммы элементов вектора
* Деление суммы на количество элементов
4. `PostProcessingImpl`
* Пост обработка результата
## 4. Схема параллелизации
Параллельный алгоритм вычисления среднего значения элементов вектора основан на распределении данных между процессами и последующей коллективной операции редукции суммы. Реализация выполняется в рамках технологии MPI с использованием механизма обмена сообщениями.

Параллельный алгоритм состоит из следующих этапов:
1. Определение параметров MPI и распределение ролей
* Каждый процеес получает `rank` и `size`
```cpp
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
```
2. Расчёт диапазонов данных для каждого процесса
* каждый процесс получает `chunk` элементов
* первые `remainder` процессов получают на один элемент больше
* диапазоны не пересекаются и покрывают весь исходный массив
```cpp
size_t chunk = total_size / size;
size_t remainder = total_size % size;

size_t start = (rank * chunk) + std::min(rank, static_cast<int>(remainder));
size_t end = start + chunk + (std::cmp_less(rank, remainder) ? 1 : 0);
```
3. Каждый процесс локально вычисляет сумму своего подмассива
```cpp
double local_sum = 0;
for (size_t i = start; i < end; ++i) {
    local_sum += vec[i];
}
```
4. Сложение всех вычислений
* После получения локальных сумм выполняется коллективная операция `MPI_Allreduce`
```cpp
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
```
5. Вычисление итогового результата
```cpp
double avg = global_sum / static_cast<double>(total_size);
GetOutput() = avg;
```

## 5. Детали реализации
### 5.1 Структура кода
Реализация параллельного алгоритма вычисления среднего значения элементов вектора расположена в каталоге mpi/:
* в mpi/include/ops_mpi.hpp - Заготовочный файл класса MPI-задачи
* в mpi/src/ops_mpi.cpp - Реализация выполнения вычисления

Класс `ZorinDAvgVecMPI` наследуется от базового класса `BaseTask`, что обеспечивает единый жизненный цикл выполнения: `ValidationImpl` → `PreProcessingImpl` → `RunImpl` → `PostProcessingImpl`.

### 5.2 Ключевые классы и функции
* `ZorinDAvgVecMPI` - Основной класс MPI-задачи
* `ValidationImpl()` - Проверка входных данных
* `PreProcessingImpl()` - Подготовительные действия перед выполнением основной логики
* `RunImpl()` - Основной этап вычисления среднего значения
* `PostProcessingImpl()` - Пост обработка результата

### 5.3 Реализация методов
####  Конструктор
```cpp
ZorinDAvgVecMPI::ZorinDAvgVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}
```
####  Валидация
```cpp
bool ZorinDAvgVecMPI::ValidationImpl() {
  return true;
}
```

#### Предварительная обработка
```cpp
bool ZorinDAvgVecMPI::PreProcessingImpl() {
  return true;
}
```

#### Основной этап
```cpp
bool ZorinDAvgVecMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &vec = GetInput();
  size_t total_size = vec.size();

  if (total_size == 0) {
    GetOutput() = 0.0;
    return true;
  }

  size_t chunk = total_size / size;
  size_t remainder = total_size % size;

  size_t start = (rank * chunk) + std::min(rank, static_cast<int>(remainder));
  size_t end = start + chunk + (std::cmp_less(rank, remainder) ? 1 : 0);

  double local_sum = 0;
  for (size_t i = start; i < end; ++i) {
    local_sum += vec[i];
  }

  double global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  double avg = global_sum / static_cast<double>(total_size);

  GetOutput() = avg;

  return true;
}
```

#### Пост обработка
```cpp
bool ZorinDAvgVecMPI::PostProcessingImpl() {
  return true;
}
```

## 6. Экспериментальная установка
### Аппаратное обеспечение/ОС
1. Модель процессора: AMD Ryzen 5 2600 (6 ядер / 12 потоков)
2. Оперативная память: 16 GB DDR4
3. версия ОС: Windows 11, 64-bit
### Набор инструментов
1. Компилятор: MSVC
2. Система сборки: CMake 
3. Тип сборки: Release
### Среда
1. PPC_NUM_PROC: 1, 2, 4, 8
- Данные: как генерируются или передаются тестовые данные (относительные пути)

### Тестовые данные
Используются входные наборы данных из каталога data/:
```
avg_vec_data.txt - основные данные
avg_vec_empty.txt - пустой файл
avg_vec_equal.txt - одинаковые элементы вектора
avg_vec_negative.txt - положительные и отрицательные элементы вектора
avg_vec_single.txt - один элемент вектора
```

## 7. Результаты и обсуждение

### 7.1 Корректность
Корректность реализации была проверена с помощью функциональных тестов (tests/functional), которые сравнивали результаты SEQ и MPI реализаций на подготовленных тестовых наборах данных:
* avg_vec_data.txt
* avg_vec_equal.txt
* avg_vec_negative.txt
* avg_vec_single.txt
* avg_vec_empty.txt

Во всех тестах результаты вычислений обеих реализаций совпали с эталонными значениями с точностью `1e-6`.

Все тесты успешно пройдены:

```log
[----------] 10 tests from AvgVecFunc/ZorinDAvgVecFuncTests
[ RUN      ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_mpi_enabled_avg_vec_data_txt
[       OK ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_mpi_enabled_avg_vec_data_txt (31 ms)
[ RUN      ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_mpi_enabled_avg_vec_equal_txt
[       OK ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_mpi_enabled_avg_vec_equal_txt (25 ms)
[ RUN      ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_mpi_enabled_avg_vec_negative_txt
[       OK ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_mpi_enabled_avg_vec_negative_txt (29 ms)
[ RUN      ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_mpi_enabled_avg_vec_single_txt
[       OK ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_mpi_enabled_avg_vec_single_txt (29 ms)
[ RUN      ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_mpi_enabled_avg_vec_empty_txt
[       OK ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_mpi_enabled_avg_vec_empty_txt (26 ms)
[ RUN      ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_seq_enabled_avg_vec_data_txt
[       OK ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_seq_enabled_avg_vec_data_txt (23 ms)
[ RUN      ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_seq_enabled_avg_vec_equal_txt
[       OK ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_seq_enabled_avg_vec_equal_txt (25 ms)
[ RUN      ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_seq_enabled_avg_vec_negative_txt
[       OK ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_seq_enabled_avg_vec_negative_txt (25 ms)
[ RUN      ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_seq_enabled_avg_vec_single_txt
[       OK ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_seq_enabled_avg_vec_single_txt (22 ms)
[ RUN      ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_seq_enabled_avg_vec_empty_txt
[       OK ] AvgVecFunc/ZorinDAvgVecFuncTests.FunctionalAverageCheck/zorin_d_avg_vec_seq_enabled_avg_vec_empty_txt (27 ms)
[----------] 10 tests from AvgVecFunc/ZorinDAvgVecFuncTests (275 ms total)
```


### 7.2 Производительность
Текущее время, ускорение и эффективность. Таблица примеров:

| Режим | Количество | Время, с | Ускорение | Эффективность |
|-------|------------|---------|---------|---------------|
| seq   | 1          | 0.827 | 1.00 | N/A           |
| mpi   | 1          | 1.155 | 0.72 | 72%           |
| mpi   | 2          | 0.654 | 1.26 | 63%          |
|mpi| 3          |0.514|1.61| 53.6%          |
|mpi| 4| 1.43| 1.92| 48.1%          |

#### Анализ производительности

* MPI-версия демонстрирует ускорение по сравнению с SEQ-реализацией начиная с 2 процессов
* На 4 процессах достигается ускорение до 2.08x
* Эффективность уменьшается при увеличении количества процессов
* Масштабируемость ограничивается отношением полезных вычислений к коммуникационным затратам

## 8. Выводы
В ходе работы были разработаны и исследованы две реализации задачи вычисления среднего значения элементов вектора: последовательная (SEQ) и параллельная (MPI). Параллельная версия обеспечивает ускорение вычислений благодаря распределению данных между несколькими процессами и коллективной операции суммирования (MPI_Allreduce).

Экспериментальные результаты показали, что MPI-реализация превосходит SEQ при использовании двух и более процессов, достигая ускорения до 2.08 раза на 4 процессах. Эффективность параллельного решения снижается при увеличении числа процессов, что объясняется низкой вычислительной сложностью задачи и накладными расходами на коммуникации.

## 9. Список литературы
1. MPI Forum. Message Passing Interface Standard. - https://www.mpi-forum.org/docs/
2. Часть 1. MPI — Введение и первая программа - https://habr.com/ru/articles/548266/
3. Часть 2. MPI — Учимся следить за процессами - https://habr.com/ru/articles/548418/

## Приложение (необязательно)
### `common.hpp`
```cpp
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace zorin_d_avg_vec {

using InType = std::vector<int>;
using OutType = double;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zorin_d_avg_vec
```

### `ops_mpi.hpp`
```cpp
#pragma once

#include "task/include/task.hpp"
#include "zorin_d_avg_vec/common/include/common.hpp"

namespace zorin_d_avg_vec {

class ZorinDAvgVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZorinDAvgVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zorin_d_avg_vec

```

### `ops_mpi.cpp`
```cpp
#include "zorin_d_avg_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "zorin_d_avg_vec/common/include/common.hpp"

namespace zorin_d_avg_vec {

ZorinDAvgVecMPI::ZorinDAvgVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ZorinDAvgVecMPI::ValidationImpl() {
  return true;
}

bool ZorinDAvgVecMPI::PreProcessingImpl() {
  return true;
}

bool ZorinDAvgVecMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &vec = GetInput();
  size_t total_size = vec.size();

  if (total_size == 0) {
    GetOutput() = 0.0;
    return true;
  }

  size_t chunk = total_size / size;
  size_t remainder = total_size % size;

  size_t start = (rank * chunk) + std::min(rank, static_cast<int>(remainder));
  size_t end = start + chunk + (std::cmp_less(rank, remainder) ? 1 : 0);

  double local_sum = 0;
  for (size_t i = start; i < end; ++i) {
    local_sum += vec[i];
  }

  double global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  double avg = global_sum / static_cast<double>(total_size);

  GetOutput() = avg;

  return true;
}

bool ZorinDAvgVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zorin_d_avg_vec

```

### `ops_seq.hpp`
```cpp
#pragma once

#include "task/include/task.hpp"
#include "zorin_d_avg_vec/common/include/common.hpp"

namespace zorin_d_avg_vec {

class ZorinDAvgVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZorinDAvgVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zorin_d_avg_vec
```

### `ops_seq.cpp`
```cpp
#include "zorin_d_avg_vec/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "util/include/util.hpp"
#include "zorin_d_avg_vec/common/include/common.hpp"

namespace zorin_d_avg_vec {

ZorinDAvgVecSEQ::ZorinDAvgVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ZorinDAvgVecSEQ::ValidationImpl() {
  return true;
}

bool ZorinDAvgVecSEQ::PreProcessingImpl() {
  return true;
}

bool ZorinDAvgVecSEQ::RunImpl() {
  const auto &vec = GetInput();
  if (vec.empty()) {
    GetOutput() = 0.0;
    return true;
  }

  const double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
  GetOutput() = sum / static_cast<double>(vec.size());

  return true;
}

bool ZorinDAvgVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zorin_d_avg_vec
```

### `functional/main.cpp`
```cpp
#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zorin_d_avg_vec/common/include/common.hpp"
#include "zorin_d_avg_vec/mpi/include/ops_mpi.hpp"
#include "zorin_d_avg_vec/seq/include/ops_seq.hpp"

namespace zorin_d_avg_vec {

class ZorinDAvgVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &info) {
    const TestType &p = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);
    std::string name = std::get<2>(p);

    for (char &c : name) {
      if (!std::isalnum(c)) {
        c = '_';
      }
    }
    const std::string &task_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);

    return task_name + "_" + name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const std::string &filename = std::get<2>(params);
    expected_ = std::get<1>(params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_zorin_d_avg_vec, filename);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("File not found: " + abs_path);
    }

    input_.clear();
    double value = 0.0;
    while (file >> value) {
      input_.push_back(value);
    }
  }

  bool CheckTestOutputData(OutType &output) final {
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (!initialized) {
      return std::fabs(output - expected_) < 1e-6;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    return std::fabs(output - expected_) < 1e-6;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_{};
  OutType expected_{};
};

namespace {

TEST_P(ZorinDAvgVecFuncTests, FunctionalAverageCheck) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParams = {{
    TestType({}, 55.0, "avg_vec_data.txt"),
    TestType({}, 5.0, "avg_vec_equal.txt"),
    TestType({}, 0.0, "avg_vec_negative.txt"),
    TestType({}, 42.0, "avg_vec_single.txt"),
    TestType({}, 0.0, "avg_vec_empty.txt"),
}};

const auto kTasks =
    std::tuple_cat(ppc::util::AddFuncTask<ZorinDAvgVecMPI, InType>(kTestParams, PPC_SETTINGS_zorin_d_avg_vec),
                   ppc::util::AddFuncTask<ZorinDAvgVecSEQ, InType>(kTestParams, PPC_SETTINGS_zorin_d_avg_vec));

INSTANTIATE_TEST_SUITE_P(AvgVecFunc, ZorinDAvgVecFuncTests, ppc::util::ExpandToValues(kTasks),
                         ZorinDAvgVecFuncTests::PrintTestParam);

}  // namespace

}  // namespace zorin_d_avg_vec
```

### `perfomance/main.cpp`
```cpp
#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zorin_d_avg_vec/common/include/common.hpp"
#include "zorin_d_avg_vec/mpi/include/ops_mpi.hpp"
#include "zorin_d_avg_vec/seq/include/ops_seq.hpp"

namespace zorin_d_avg_vec {

class ZorinDAvgVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<ppc::util::FuncTestParam<InType, OutType, TestType>> &info) {
    const TestType &p = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);
    std::string name = std::get<2>(p);

    for (char &c : name) {
      if (!std::isalnum(c)) {
        c = '_';
      }
    }
    const std::string &task_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);

    return task_name + "_" + name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const std::string &filename = std::get<2>(params);
    expected_ = std::get<1>(params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_zorin_d_avg_vec, filename);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("File not found: " + abs_path);
    }

    input_.clear();
    double value = 0.0;
    while (file >> value) {
      input_.push_back(value);
    }
  }

  bool CheckTestOutputData(OutType &output) final {
    int initialized = 0;
    MPI_Initialized(&initialized);

    if (!initialized) {
      return std::fabs(output - expected_) < 1e-6;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    return std::fabs(output - expected_) < 1e-6;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_{};
  OutType expected_{};
};

namespace {

TEST_P(ZorinDAvgVecFuncTests, FunctionalAverageCheck) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParams = {{
    TestType({}, 55.0, "avg_vec_data.txt"),
    TestType({}, 5.0, "avg_vec_equal.txt"),
    TestType({}, 0.0, "avg_vec_negative.txt"),
    TestType({}, 42.0, "avg_vec_single.txt"),
    TestType({}, 0.0, "avg_vec_empty.txt"),
}};

const auto kTasks =
    std::tuple_cat(ppc::util::AddFuncTask<ZorinDAvgVecMPI, InType>(kTestParams, PPC_SETTINGS_zorin_d_avg_vec),
                   ppc::util::AddFuncTask<ZorinDAvgVecSEQ, InType>(kTestParams, PPC_SETTINGS_zorin_d_avg_vec));

INSTANTIATE_TEST_SUITE_P(AvgVecFunc, ZorinDAvgVecFuncTests, ppc::util::ExpandToValues(kTasks),
                         ZorinDAvgVecFuncTests::PrintTestParam);

}  // namespace

}  // namespace zorin_d_avg_vec
```
