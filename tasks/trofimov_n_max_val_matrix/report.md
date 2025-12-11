# Поиск максимальных значений по строкам матрицы

* Студент: Трофимов Никита, группа 3823Б1ПР4
* Технология: SEQ | MPI
* Вариант: 15
* Преподаватель: Сысоев Александр Владимирович, лектор, доцент кафедры высокопроизводительных вычислений и системного программирования

## 1. Введение

Задача поиска максимальных элементов по строкам матрицы является фундаментальной операцией в линейной алгебре и обработке данных. При работе с большими матрицами последовательные алгоритмы могут становиться узким местом в вычислительных системах. Использование параллельных вычислений с технологией MPI позволяет эффективно распределить обработку строк матрицы между несколькими процессами.

В данной работе реализованы две версии алгоритма:

* Последовательная (SEQ) – для определения базовой производительности
* Параллельная (MPI) – с распределением строк матрицы между процессами

Цель работы: разработать эффективную параллельную реализацию и измерить достигнутое ускорение на различном количестве процессов.

## 2. Постановка задачи

**Входные данные:**

* Матрица размером M×N, где M, N > 0

**Выходные данные:**

* Вектор размером M, содержащий максимальные элементы каждой строки

**Формат данных:**

* Матрица генерируется детерминированно для тестирования
* Элементы матрицы: целые числа
* Для тестов производительности используется матрица 5000×5000

**Ограничения:**

* M > 0, N > 0
* Все строки матрицы имеют одинаковую длину
* Выходной вектор должен быть пуст перед выполнением

## 3. Базовый алгоритм (последовательный)

Последовательный алгоритм выполняет проход по всем строкам матрицы:

```
1. Инициализировать выходной вектор result размером M
2. Для каждой строки i от 0 до M-1:
   a. Инициализировать max_val = первый элемент строки
   b. Для каждого элемента j в строке i:
      - Если matrix[i][j] > max_val, то max_val = matrix[i][j]
   c. result[i] = max_val
3. Вернуть result
```

**Сложность:**

* Временная: O(M×N)
* Пространственная: O(M)

## 4. Схема параллелизации

### Распределение данных

Используется блочная декомпозиция по строкам матрицы. Каждый процесс обрабатывает свой набор строк.

Для P процессов и M строк:

```
rows_per_process = M / P
remainder = M % P

start_row[rank] = rank * rows_per_process + min(rank, remainder)
local_rows[rank] = rows_per_process + (rank < remainder ? 1 : 0)
```

Первые `remainder` процессов получают на 1 строку больше для равномерного распределения.

### Коммуникация

1. Локальные вычисления: каждый процесс находит максимумы в своих строках
2. `MPI_Gather`: сбор информации о количестве строк каждого процесса
3. `MPI_Gatherv`: сбор локальных результатов на процессе 0 с формированием итогового вектора

### Роли процессов

* **Процесс 0 (Root)**: собирает результаты от всех процессов и формирует итоговый вектор
* **Все процессы**: обрабатывают свой блок строк и участвуют в коллективных операциях

## 5. Детали реализации

### Структура кода

```
trofimov_n_max_val_matrix/
├── common/include/common.hpp
├── mpi/include/ops_mpi.hpp
├── mpi/src/ops_mpi.cpp
├── seq/include/ops_seq.hpp
├── seq/src/ops_seq.cpp
└── tests/
    ├── functional/main.cpp
    └── performance/main.cpp
```

### Классы

* `TrofimovNMaxValMatrixSEQ` – последовательная реализация
* `TrofimovNMaxValMatrixMPI` – параллельная реализация

### Методы

* `ValidationImpl()` – проверка корректности матрицы
* `PreProcessingImpl()` – инициализация выходного вектора
* `RunImpl()` – основной алгоритм поиска максимумов
* `PostProcessingImpl()` – проверка результатов

### Особенности реализации

* Использование `std::ranges::max_element` для поиска максимума
* Равномерное распределение строк при неравномерном делении
* Минимальные коммуникации между процессами

## 6. Экспериментальная установка

### Аппаратное обеспечение

* Процессор: AMD Ryzen 7 7700
* Оперативная память: 16 ГБ
* Операционная система: Windows

### Программное обеспечение

* Компилятор: MSVC
* MPI: MS-MPI
* Система сборки: CMake
* Тестовый фреймворк: Google Test

### Параметры тестирования

**Функциональные тесты:**

* Матрицы различных размеров
* Проверка граничных случаев
* Сравнение SEQ и MPI результатов

**Тесты производительности:**

* Матрица: 5000×5000 элементов
* Количество процессов: 1, 2, 4
* Режимы: task_run, pipeline

## 7. Результаты и обсуждение

### 7.1 Корректность

* Все функциональные тесты успешно пройдены для MPI версии и SEQ task_run
* SEQ pipeline версия показывает аномально большое время выполнения (91874+ сек)

### 7.2 Производительность

| Процессы | Режим    | Версия | Время, сек | Ускорение | Эффективность |
| -------- | -------- | ------ | ---------- | --------- | ------------- |
| 1        | task_run | SEQ    | 0.01041    | 1.00      | N/A           |
| 1        | task_run | MPI    | 0.01100    | 0.95      | 95.0%         |
| 1        | pipeline | MPI    | 0.01170    | 0.89      | 89.0%         |
| 2        | task_run | MPI    | 0.00819    | 1.27      | 63.5%         |
| 2        | pipeline | MPI    | 0.00838    | 1.24      | 62.0%         |
| 4        | task_run | MPI    | 0.00368    | 2.83      | 70.8%         |
| 4        | pipeline | MPI    | 0.00363    | 2.87      | 71.8%         |

**Расчет показателей:**

* Ускорение = T_seq_task_run / T_mpi
* Эффективность = (Ускорение / P) × 100%

**Анализ:**

* На 1 процессе MPI: накладные расходы 5–11%
* На 2 процессах: ускорение 1.24–1.27×, эффективность 62–63.5%
* На 4 процессах: ускорение 2.83–2.87×, эффективность 71–72%
* Режимы task_run и pipeline показывают схожую производительность

## 8. Заключение

* Разработаны последовательная и параллельная версии алгоритма
* MPI версия и SEQ task_run проходят все тесты
* На матрице 5000×5000 достигнуто максимальное ускорение 2.87× на 4 процессах
* Эффективность параллельной версии удовлетворительная
* Рекомендации: исправить SEQ pipeline версию, оптимизировать коммуникации, исследовать влияние размера матрицы, рассмотреть асинхронные операции

## 9. Источники

1. MPI Forum. *MPI: A Message-Passing Interface Standard*
2. Gropp W., Lusk E., Skjellum A. *Using MPI: Portable Parallel Programming with the Message-Passing Interface*
3. Microsoft. *Microsoft MPI Documentation*
4. Сысоев А. В. *Лекции по параллельному программированию*

## Приложение

### Последовательная версия (SEQ)

**RunImpl() - основная логика:**

```cpp
bool TrofimovNMaxValMatrixSEQ::RunImpl() {
  const auto &input = GetInput();

  for (std::size_t i = 0; i < input.size(); ++i) {
    if (!input[i].empty()) {
      GetOutput()[i] = *std::ranges::max_element(input[i]);
    }
  }

  return true;
}
```

### Параллельная версия (MPI)

**RunImpl() - основная логика:**

```cpp
bool TrofimovNMaxValMatrixMPI::RunImpl() {
  const auto &input = GetInput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto rows = static_cast<int>(input.size());
  const int rows_per_process = rows / size;
  const int remainder = rows % size;

  const int start_row = (rank * rows_per_process) + std::min(rank, remainder);
  const int end_row = ((rank + 1) * rows_per_process) + std::min(rank + 1, remainder);
  const int local_rows = end_row - start_row;

  std::vector<int> local_maxima(static_cast<std::size_t>(local_rows));
  for (int i = 0; i < local_rows; ++i) {
    const int global_row_index = start_row + i;
    local_maxima[static_cast<std::size_t>(i)] =
        *std::ranges::max_element(input[static_cast<std::size_t>(global_row_index)]);
  }

  std::vector<int> recv_counts(static_cast<std::size_t>(size));
  std::vector<int> displacements(static_cast<std::size_t>(size));

  MPI_Gather(&local_rows, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, kRootRank, MPI_COMM_WORLD);

  if (rank == kRootRank) {
    displacements[0] = 0;
    for (int i = 1; i < size; ++i) {
      displacements[static_cast<std::size_t>(i)] =
          displacements[static_cast<std::size_t>(i - 1)] + recv_counts[static_cast<std::size_t>(i - 1)];
    }
  }

  if (rank == kRootRank) {
    GetOutput().resize(static_cast<std::size_t>(rows));
  }

  MPI_Gatherv(local_maxima.data(), local_rows, MPI_INT, GetOutput().data(), 
              recv_counts.data(), displacements.data(), MPI_INT, kRootRank, MPI_COMM_WORLD);

  return true;
}
```

### Формула распределения данных

```cpp
int rows_per_process = M / P;
int remainder = M % P;
int start_row = rank * rows_per_process + min(rank, remainder);
int local_rows = rows_per_process + (rank < remainder ? 1 : 0);
```
