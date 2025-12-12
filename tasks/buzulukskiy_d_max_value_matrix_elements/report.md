- Student: <Бузулукский Данила Сергеевич>, group <3823Б1ПР5>
- Technology: <SEQ | MPI>
- Variant: <13>

## 1. Introduction
МОТИВАЦИЯ -> Ускорение поиска максимально элемента в матрицах большого размера,за счёт распределения нагрузки по нескольким процесам
ПРОБЛЕМА  -> Матрицы большого размера могут очень долго обрабатываться
РЕЗУЛЬТАТ -> Ускорение производительности.

## 2. Problem Statement
Formal task definition -> для матрицы размером A[a1,b1] найти максимальный элемент
input/output format    -> на вход подаётся Matrix(тоесть данные матрицы), на выход подаётся 1 int значение, которое является максимумом в матрице
constraints            -> a1,b1 > 0, matrix.size() = a1 * b1.

## 3. Baseline Algorithm (Sequential)
Describe the base algorithm with enough detail to reproduce.

```cpp
int max_value = matrix[0];
for (size_t i = 1; i < matrix.size(); i++) {
    max_value = std::max(max_value, matrix[i]);
  }
  GetOutput() = max_value;
```

## 4. Parallelization Scheme
data distribution:
Блочное распределение по строкам
Балансировка нагрузки при неравномерном распределении
, communication pattern:
```cpp
//распределение данных
MPI_Scatterv(matrix.data(), how_many_to_one_proces.data(), offset.data(), MPI_INT, recvbuf.data(),
               how_many_to_one_proces[rank], MPI_INT, 0, MPI_COMM_WORLD);
//поиск локального максимума
  int local_max;
  if (!recvbuf.empty()) {
    local_max = recvbuf[0];
    for (size_t i = 1; i < recvbuf.size(); i++) {
      local_max = std::max(local_max, recvbuf[i]);
    }
  } else {
    local_max = 0;
  }
//поиск глобального максимума
  int global_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
```
 rank roles.
 Rank 0 - распределение задач
 Rank all - локальные вычисления и участие в редукции


## 5. Implementation Details
- Code structure (files, key classes/functions)
common - общие структуры данных
mpi - паралельная реализация mpi
seq - последовательная
test - тесты функцианальности и производительности

```cpp
class BuzulukskiyDMaxValueMatrixElementsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BuzulukskiyDMaxValueMatrixElementsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
```
```cpp
class BuzulukskiyDMaxValueMatrixElementsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BuzulukskiyDMaxValueMatrixElementsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

```
```cpp
struct Matrix {
  std::vector<int> data;
  int rows;
  int columns;
};
```
- Important assumptions and corner cases
 1- матрица не должна быть пустой
 2- размер матрицы должен соответствовать rows*columns
 3- возращает 1 элемент(максимально значение в матрице)
- Memory usage considerations
 1- в mpi каждый процесс имеет только свои значения


## 6. Experimental Setup
- Процессор: ryzen 5 5600x
- Количество ядер: 6
- Количество потоков: 12
- ОЗУ: 32 Гб
- ОС: Windows 10
- Архитектура: x64

- Язык программирования: C++
- Библиотека для параллельного программирования: MPI
- Компилятор MSCV
- Тип сборки: Release 

## 7. Results and Discussion

### 7.1 Correctness
Размер матрицы был 4000x4000
Было произведено 4 запуска
В каждом из запусков MPI оказалось эффективнее SEQ

### 7.2 Performance
Present time, speedup and efficiency. Example table:

| Mode        | processes | AvgTime(s) | Speedup | Efficiency |
|-------------|-----------|------------|---------|------------|
| seq         | 1         | 0.09386    | 1.00    | N/A        |
| mpi         | 2         | 0.06325    | 1.48    | 74.0%      |
| mpi         | 4         | 0.04038    | 2.32    | 58.0%      |
| mpi         | 6         | 0.04205    | 2.23    | 37.2%      |
| mpi         | 8         | 0.03589    | 2.62    | 32.7%      |


## 8. Conclusions
ВЫВОД :использование mpi показало свою эффективность, ускорив работу в 2.32 раза в среднем
таким образом можно сделать вывод, что распределение нагрузки на несколько процессов является
максимально эффективным метод работы с большим количеством данных
Также можно заметить, что на 6 процессах время выполнения больше чем на 4, это может быть связано
с тем, что распределение проверяемой матрицы получилось не равномерным.
Наибольшая эффективность достигается на 2 процессах, однако максимальное абсолютное ускорение на 8 процессах
Ограничения: если матрицы маленькие, то mpi является неэффективным методом работы
Следовательно использование MPI является эффективным в условиях больших матриц когда комуникация между процессами
не затратит времени больше, чем сама работа с данными

## 9. References
  MICROSOFT MPI - https://learn.microsoft.com/ru-ru/message-passing-interface/microsoft-mpi
  Parallel Programming Course - https://learning-process.github.io/parallel_programming_course/ru/index.html
  Parallel Programming 2025-2026 - https://disk.yandex.ru/d/NvHFyhOJCQU65w
  stack overflow - https://stackoverflow.com/questions
