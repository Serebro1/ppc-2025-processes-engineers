# Вычисление среднего значения элементов вектора

- Студент: Отческов Семён Андреевич, группа 3823Б1ПР1
- Технологии: SEQ | MPI
- Вариант: 2

## 1. Введение
- Вычисление среднего арифметического элементов вектора является хоть и простой но важной задачей в анализе данных. При работе с большими объемами данных (несколько миллионов или даже миллиардов элементов) скорость вычислений простого алгоритма может оказаться недостаточной даже при использовании оптимизаций компилятора.

- В данной работе представлены два алгоритма вычисления среднего значения элементов вектора: последовательная (базовая) реализация и параллельная реализация с использованием технологии MPI (Message Passing Interface).

- **Цель работы:** сравнение производительности алгоритмов и анализ эффективности распараллеливания вычислительной задачи.

## 2. Постановка задачи
**Формальная постановка:**
- Для вектора V длины N вычислить среднее арифметическое: $avg=\frac{\sum_{i=0}^{N-1}V[i]}{N}$

**Входные данные:**
- Вектор целых чисел произвольной длины N.

**Выходные данные:**
- Вещественное число — среднее значение элементов вектора V.

**Ограничения:**
- Вектор должен содержать хотя бы один элемент.
- Вектор содержит целые числа.

## 3. Описание алгоритма (последовательного)

### 3.1. Этапы выполнения задачи
**1. Валидация данных (`ValidationImpl`):**
- Проверка на пустоту вектора.
- Проверка корректности начального состояния выходного значения.

**2. Предобработка данных (`PreProcessingImpl`):**
- Задача не требует предобработки, поэтому данный этап пропускается.

**3. Вычисления (`RunImpl`):**
- Подсчёт суммы элементов вектора.
- Деление полученой суммы на число элементов в векторе.

**4. Постобработка данных (`PostProcessingImpl`):**
- Аналогична предобработке.

### 3.2. Сложность алгоритма:
- Временная сложность: `O(N)` — однократный проход по каждому элементу вектора.
- Пространственная сложность: `O(N)` — хранение вектора произвольной длины N.

## 4. Схема распараллеливания алгоритма с помощью MPI

### 4.1. Распределение данных
- Блочное распределение:
  - Размер базового блока (`batch_size`) элементов определяется как `число элементов вектора / число процессов`.
  - Размер локальной части для процесса (`proc_size`) равно размеру базового блока.
  - Если есть остаток от деления числа элементов, то он добавляется в `proc_size` последнего процесса.
  - Начало сегмента данных для процесса определяется как `начало исходного вектора + ранг процесса * размер базового блока`.
  - Конец сегмента данных определяется как `начало локальной части + proc_size`.
- Таким образом, каждый процесс работает с помощью итераторов напрямую со своей локальной частью данных в исходном векторе.

### 4.2. Топология коммуникаций
- Линейная топология (все процессы связаны через MPI_COMM_WORLD).
- Все процессы равноправны — отсутствие выделенных master/slave процессов.
- Все процессы участвуют в коммуникации.

### 4.3. Паттерны коммуникации
- В MPI реализации используется коллективная коммуникация «Каждый к каждому».
- Коллективная функция `MPI_Allreduce`:
    ```c++
    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ```
- Характеристики `MPI_Allreduce`:
  - Тип операции: All-to-All с агрегацией результатов.
  - Режим работы: блокирующая операция.
  - Операция: суммирование (`MPI_SUM`).
  - Объём передаваемых данных: одно целое число с каждого процесса.

### 4.4. Распределение вычислений
1. **Инициализация MPI** — получение ранга и числа процессов.
2. **Определение границ** — вычисление локального сегмента данных.
3. **Локальные вычисления**— суммирование элементов локального сегмента.
4. **Глобальная редукция** — объединение локальных сумм.
5. **Финальное вычисление** — расчёт среднего значения.

## 5. Особенности реализаций

### 5.1. Структура кода
Реализации классов и методов на языке С++ указаны в [Приложении](#10-приложение).

#### 5.1.1. Файлы
- `./common/include/common.hpp` — общие определения типов данных ([см. Приложение №1](#101-приложение-1--общие-определения)).
- `./seq/include/ops_seq.hpp` — определение класса последовательной версии задачи ([см. Приложение №2.1](#1021-заголовочный-файл)).
- `./seq/include/ops_mpi.hpp` — определение класса параллельной версии задачи ([см. Приложение №2.2](#1021-файл-реализации)).
- `./seq/src/ops_seq.cpp` — реализация последовательной версии задачи ([см. Приложение №3.1](#1031-заголовочный-файл)).
- `./seq/src/ops_mpi.cpp` — реализация параллельной версии задачи ([см. Приложение №3.2](#1032-файл-реализации)).
- `./tests/functional/main.cpp` — реализация функциональных и валидационных тестов ([см. Приложение №4](#104-приложение-4--функциональные-и-валидационные-тесты)).
- `./tests/performance/main.cpp` — реализация производительных тестов ([см. Приложение №5](#105-приложение-5--проиводительные-тесты)).

#### 5.1.2. Ключевые классы
- `OtcheskovSElemVecAvgSEQ` — последовательная версия.
- `OtcheskovSElemVecAvgMPI` — параллельная версия.
- `OtcheskovSElemVecAvgFuncTests` — функциональные тесты.
- `OtcheskovSElemVecAvgFuncTestsValidation` — валидационные тесты.
- `OtcheskovSElemVecAvgPerfTests` — производительные тесты.

#### 5.1.3. Основные методы
- `ValidationImpl` — валидация входных данных и состояния выходных данных.
- `PreProcessingImpl` — препроцессинг, не используется.
- `RunImpl` — основная логика вычислений.
- `PostProcessingImpl` — постпроцессинг, не используется.

### 5.2. Реализация последовательной версии

#### 5.2.1. ValidationImpl
- Проверка на пустоту вектора `GetInput` и начального значения выходного параметра `GetOutput`.
- Выходное значение `GetOutput` инициализируется значением `NAN` для однозначной идентификации начального состояния.
- `NAN` - макрос, который раскрывается в константное выражение типа float, представляющее значение тихого NaN (QNaN).
- Проверка, что `GetOutput` не был изменён до начала задачи, выполняется функцией `std::isnan` из библиотеки `cmath`.

**Реализация на C++:**
```c++
bool OtcheskovSElemVecAvgMPI::ValidationImpll() {
  return (!GetInput().empty() && std::isnan(GetOutput()));
}
```

#### 5.2.2. PreProcessingImpl
В препроцессинге нет необходимости, поэтому данный этап пропускается.

**Реализация на C++:**
```c++
bool OtcheskovSElemVecAvgSEQ::PreProcessingImpl() {
  return true;
}
```

#### 5.2.3. RunImpl
- Дополнительная проверка на пустоту входного вектора.
- Сумма элементов массива вычисляется STL-функцией `std::reduce`, расположенной в библиотеке `<numeric>`.
- Функция `std::reduce` принимает пару итераторов, определяющих диапазон элементов.
  - Она складывает числа в произвольном порядке, предоставляя компилятору больше свободы для оптимизации.
- Присвоение `GetOutput` среднего арифметрического, поделив полученную сумму на размер входного вектора.
- Проверяем, что `GetOutput` был изменён в результате предыдущей операции с помощью `!std::isnan`.

**Реализация на C++:**
```c++
bool OtcheskovSElemVecAvgSEQ::RunImpl() {
  // проверка на пустоту вектора
  if (GetInput().empty()) {
    return false;
  }

  // вычисляем среднее арифметическое элементов вектора
  int64_t sum = std::reduce(GetInput().begin(), GetInput().end(), static_cast<int64_t>(0));
  GetOutput() = static_cast<double>(sum) / static_cast<double>(GetInput().size());
  return !std::isnan(GetOutput());
}
```

#### 5.2.4. PostProcessingImpl
В постпроцессинге нет необходимости, поэтому данный этап пропускается.

**Реализация на C++:**
```c++
bool OtcheskovSElemVecAvgSEQ::PostProcessingImpl() {
  return true;
}
```

### 5.3. Реализация параллельной версии
Этапы: **ValidationImpl**, **PreProcessingImpl**, **PostProcessingImpl** аналогичны последовательной версии.

#### 5.3.1. RunImpl
- Дополнительная проверка на пустоту вектора.
- Распределение данных, как описано в разделе [4.1. Распределение данных](#41-распределение-данных).
- Вычисление локальных сумм.
- С помощью `MPI_Allreduce` выполняется:
  - Сбор локальных сумм из процессов.
  - Сложение полученных сумм.
  - Передача результата всем процессам.

**Реализация на C++:**
```c++
bool OtcheskovSElemVecAvgMPI::RunImpl() {
  // проверка на пустоту вектора
  if (GetInput().empty()) {
    return false;
  }

  // ранг процесса и число процессов
  int proc_rank{};
  int proc_num{};
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  // распределение данных процессам
  const size_t total_size = GetInput().size();
  const size_t batch_size = total_size / proc_num;
  const size_t proc_size = batch_size + (proc_rank == proc_num - 1 ?total_size % proc_num : 0);
  
  // процесс определяет свои границы локальных данных
  auto start_local_data = GetInput().begin() + static_cast<std::vector<int>::difference_type>(proc_rank * batch_size);
  auto end_local_data = start_local_data + static_cast<std::vector<int>::difference_type>(proc_size);

  // процесс вычисляет свою локальную сумму
  int64_t local_sum = std::reduce(start_local_data, end_local_data, static_cast<int64_t>(0));
  int64_t total_sum = 0;
  
  // собирает у процессов локальные суммы, складывает их в total_sum и раздёт процессам
  MPI_Allreduce(&local_sum, &total_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  
  // вычисляет среднее арифметическое
  GetOutput() = static_cast<double>(total_sum) / static_cast<double>(total_size);
  return !std::isnan(GetOutput());
}
```

### 5.4. Использование памяти
- Последовательная версия: `O(N)` — входной вектор произвольной длины N.
- Параллельная версия: `O(N * число процессов)` — каждый процесс хранит полную копию вектора.

### 5.5. Допущения и крайние случаи
- Не вполне равномерное распределение:
  - На малом числе процессов не влияет на производительность.
  - Минимизирует код для распределения данных.

- В параллельной версии каждый процесс хранит входной вектор:
  - Требует больше оперативной памяти для хранения копии вектора в каждом процессе.
  - Минимизирует число необходимых коммуникаций между процессами.


## 6. Тестовые инфраструктуры
### 6.1. Windows
| Параметр   | Значение                                             |
| ---------- | ---------------------------------------------------- |
| CPU        | Intel Core i5 12400F (6 cores, 12 threads, 2500 MHz) |
| RAM        | 32 GB DDR4 (3200 MHz)                                |
| OS         | Windows 10 (10.0.19045)                              |
| Компилятор | MSVC 19.42.34435, Release Build                      |

### 6.2 WSL
| Параметр   | Значение                                             |
| ---------- | ---------------------------------------------------- |
| CPU        | Intel Core i5 12400F (6 cores, 12 threads, 2500 MHz) |
| RAM        | 16 GB DDR4 (3200 MHz)                                |
| OS         | Ubuntu 24.04.3 LTS on Windows 10 x86_64              |
| Компилятор | GCC 13.3.0, Release Build                            |

### 6.3. Общие настройки
- **Переменные окружения:** PPC_NUM_PROC = 2, 4.
- **Данные:** элементы вектора хранятся в `test_vec*.txt` файлах в директории `./data`.

## 7. Результаты и обсуждение

### 7.1. Корректность

Корректность задачи для обоих версий проверена с помощью набора параметризированных тестов Google Test.

#### 7.1.1. Функциональные тесты
- Вектор с положительными числами:
  - Данные: `./data/test_vec1.txt`.
  - Ожидаемое значение: `50.5`.

- Вектор со смешанными положительными и отрицательными числами:
  - Данные: `./data/test_vec2.txt`.
  - Ожидаемое значение: `14.5`.

- Вектор с одним элементом:
  - Данные: `./data/test_vec_one_elem.txt`.
  - Ожидаемое значение: `5.0`.

- Вектор с дробным средним значением элементов:
  - Данные: `./data/test_vec_fraction.txt`.
  - Ожидаемое значение: `4.0/3.0`.

- Вектор с большого размера:
  - Данные: `./data/test_vec_one_million_elems.txt`.
  - Ожидаемое значение: `-2.60988`.

- Вектор с чередующимися противоположными элементами:
  - Данные: `./data/test_vec_alternating_elems.txt`.
  - Ожидаемое значение: `0.0`.

- Вектор с нулевыми элементами:
  - Данные: `./data/test_vec_zeros_elems.txt`.
  - Ожидаемое значение: `0.0`.

#### 7.1.2. Валидационные тесты
- Обработка пустого вектора:
  - Данные: пустой вектор.
  - Цель: проверка корректной обработки некорректных входных данных.

- Проверка сброса выходного значения перед выполнением:
  - Данные: любой вектор.
  - Цель: проверка провала валидации при изменении состояния переменных до запуска задачи.

#### 7.1.3. Механизм проверки
- Все тесты выполняются как для последовательной (SEQ), так и для параллельной (MPI) версии.
- Данные загружаются из файлов через абсолютные пути, получаемые функцией `GetAbsoluteTaskPath()`.
- Выходное и ожидаемое вещественные значения сравниваются с учётом машинной точности:
    ```с++
    std::fabs(expected_avg_ - output_data) < std::numeric_limits<double>::epsilon()
    ```
    где:
    - `std:fabs()` — вычисляет абсолютное значение разности ожидаемого и полученного значений.
    - `std::numeric_limits<double>::epsilon()` — возвращает машинный эпсилон для типа double.

- Для некорректных сценариев проверяется провал валидации (`ValidationImpl()`).

### 7.2. Производительные тесты

#### 7.2.1. Методология тестирования
- **Данные:** вектор из 1 миллиона элементов дублируется до 32 и 512 миллионов. Данные для вектора берутся из файла `./data/test_vec_one_million_elems.txt`.
- **Режимы:**
  - **pipeline** — запуск и измерение времени всех этапов алгоритма (`Validation -> PreProcessing -> Run -> PostProcessing`).
  - **task_run** — запуск всех этапов алгоритма, но измеряется время только на этапе `Run`.
- **Метрики:** число процессов, абсолютное время выполнения pipeline и task_run, ускорение, эффективность.
- В итоговой реализации показан тест на векторе из 512 миллионов элементов.

#### 7.2.2. Результаты тестирования на векторе из 32 миллионов элементов

**Windows:**
| Режим          | Процессов | Время, s | Ускорение | Эффективность |
| -------------- | --------- | -------- | --------- | ------------- |
| seq (pipeline) | 1         | 0.049592 | 1.0000    | N/A           |
| mpi (pipeline) | 2         | 0.024586 | 2.0159    | 100.8%        |
| mpi (pipeline) | 4         | 0.017337 | 2.8605    | 71.5%         |
| seq (task_run) | 1         | 0.049799 | 1.0000    | N/A           |
| mpi (task_run) | 2         | 0.027084 | 1.8387    | 91.9%         |
| mpi (task_run) | 4         | 0.014524 | 3.4287    | 85.7%         |

**WSL:**
| Режим          | Процессов | Время, s | Ускорение | Эффективность |
| -------------- | --------- | -------- | --------- | ------------- |
| seq (pipeline) | 1         | 0.154427 | 1.0000    | N/A           |
| mpi (pipeline) | 2         | 0.077826 | 1.9843    | 99.2%         |
| mpi (pipeline) | 4         | 0.041513 | 3.7199    | 92.9%         |
| seq (task_run) | 1         | 0.162249 | 1.0000    | N/A           |
| mpi (task_run) | 2         | 0.086869 | 1.8677    | 93.4%         |
| mpi (task_run) | 4         | 0.048871 | 3.3199    | 82.9%         |

**\*GitHub:**
| Режим          | Процессов | Время, s | Ускорение | Эффективность |
| -------------- | --------- | -------- | --------- | ------------- |
| seq (pipeline) | 1         | 0.005423 | 1.0000    | N/A           |
| mpi (pipeline) | 2         | 0.003351 | 1.6183    | 80.9%         |
| seq (task_run) | 1         | 0.005457 | 1.0000    | N/A           |
| mpi (task_run) | 2         | 0.003553 | 1.5359    | 76.8%         |

#### 7.2.3. Результаты тестирования на 512 миллионов элементов

**Windows:**
| Режим          | Процессов | Время, s | Ускорение | Эффективность |
| -------------- | --------- | -------- | --------- | ------------- |
| seq (pipeline) | 1         | 0.675529 | 1.0000    | N/A           |
| mpi (pipeline) | 2         | 0.390001 | 1.7321    | 86.6%         |
| mpi (pipeline) | 4         | 0.235353 | 2.8702    | 71.8%         |
| seq (task_run) | 1         | 0.710808 | 1.0000    | N/A           |
| mpi (task_run) | 2         | 0.388543 | 1.8294    | 91.3%         |
| mpi (task_run) | 4         | 0.235275 | 3.0212    | 75.5%         |

**WSL:**
| Режим          | Процессов | Время, s | Ускорение | Эффективность |
| -------------- | --------- | -------- | --------- | ------------- |
| seq (pipeline) | 1         | 2.443846 | 1.0000    | N/A           |
| mpi (pipeline) | 2         | 1.270955 | 1.9228    | 96.1%         |
| seq (task_run) | 1         | 2.448493 | 1.0000    | N/A           |
| mpi (task_run) | 2         | 1.248421 | 1.9613    | 98.1%         |

**\*GitHub:**
| Режим          | Процессов | Время, s | Ускорение | Эффективность |
| -------------- | --------- | -------- | --------- | ------------- |
| seq (pipeline) | 1         | 0.090900 | 1.0000    | N/A           |
| mpi (pipeline) | 2         | 0.054742 | 1.6605    | 83.0%         |
| seq (task_run) | 1         | 0.096574 | 1.0000    | N/A           |
| mpi (task_run) | 2         | 0.059635 | 1.6194    | 80.9%         |

*\*Результаты собирались на локальном форке из Github Actions*

### 7.3. Анализ результатов

- **Эффективность:**
  - Высокая эффективность на малом числе процессов (до 90%+ на двух процессах).
  - Снижение эффективности при увеличении числа процессов (до 70-85% на четырёх процессах).

- **Производительность:**
  - MPI-версия демонстрирует близкое к линейному ускорение на двух процессах.
  - Наиболее стабильной из тестовых инфраструктур является машина на GitHub.
  - WSL также показывает стабильный результат, но является самой медленной инфраструктурой, что, вероятно, связано архитектурными особенностями технологии.
  - Наименее стабильной была тестовая инфраструктура на Windows (процент эффективности на тестах варьировался от 60% до 100% при каждом запуске).

- **Ограничения масштабируемости:**
  - Требуется значительный объём оперативной памяти для хранения вектора в каждом процессе.
  - Из-за затрат на коммуникацию между процессами эффективность снижается при увеличении числа процессов.

## 8. Заключения

### 8.1. Достигнутые результаты:
1. **Эффективное распараллеливание** — достигнуто значительное ускорение на больших объёмах данных.
2. **Схема распределения данных** — блочная схема, требующая минимумального количества операций для данной задачи.
3. **Минимальные коммуникации** — использование единственной коллективной операции `MPI_Allreduce`.
4. **Корректность результатов** — полное соответствие последовательной и параллельной версий.

### 8.2. Возможные улучшения:
1. **Оптимизация использования памяти** — текущая реализация требует хранения копии исходного вектора в каждом процессе.
2. **Равномерное распределение** — устранение неравномерности при некратном делении.
3. **Улучшение масштабируемости** — снижение эффективности при большом числе процессов.

В рамках данной работы успешно решена задача вычисления среднего арифметического элементов вектора, реализованы два решения: последовательное и параллельное с использованием MPI.

Параллельное решение демонстрирует значительное ускорение и хорошую эффективность для данной задачи, что позволяет рекомендовать его использование в более сложных вычислительных задачах и при анализе данных. 

## 9. Источники
1. std::reduce // cppreference.com URL: https://en.cppreference.com/w/cpp/algorithm/reduce.html (дата обращения: 12.11.2025).
2. Документация по курсу «Параллельное программирование» // Parallel Programming Course URL: https://learning-process.github.io/parallel_programming_course/ru/index.html (дата обращения: 25.10.2025).
3. The big STL Algorithms tutorial: reduce operations // Sandor Dargo's Blog URL: https://www.sandordargo.com/blog/2021/10/20/stl-alogorithms-tutorial-part-27-reduce-operations (дата обращения: 12.11.2025).
4. "Коллективные и парные взаимодействия" / Сысоев А. В // Лекции по дисциплине «Параллельное программирование для кластерных систем».

## 10. Приложение

### 10.1. Приложение №1 — Общие определения
Файл: `./common/include/common.hpp`.
```cpp
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace otcheskov_s_elem_vec_avg {

using InType = std::vector<int>;
using OutType = double;
using TestType = std::tuple<std::string, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace otcheskov_s_elem_vec_avg
```

### 10.2. Приложение №2 — Последовательная версия решения задачи 
#### 10.2.1. Заголовочный файл:
Файл: `./seq/ops_seq.hpp`.
```cpp
#pragma once

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "task/include/task.hpp"

namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OtcheskovSElemVecAvgSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace otcheskov_s_elem_vec_avg
```

#### 10.2.1. Файл реализации:
Файл: `./seq/ops_seq.cpp`.
```cpp
#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstdint>
#include <numeric>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"

namespace otcheskov_s_elem_vec_avg {

OtcheskovSElemVecAvgSEQ::OtcheskovSElemVecAvgSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = NAN;
}

bool OtcheskovSElemVecAvgSEQ::ValidationImpl() {
  return (!GetInput().empty() && std::isnan(GetOutput()));
}

bool OtcheskovSElemVecAvgSEQ::PreProcessingImpl() {
  return true;
}

bool OtcheskovSElemVecAvgSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int sum = std::reduce(GetInput().begin(), GetInput().end());
  GetOutput() = sum / static_cast<double>(GetInput().size());
  return !std::isnan(GetOutput());
}

bool OtcheskovSElemVecAvgSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg
```

### 10.3. Приложение №3 — Параллельная версия решения задачи

#### 10.3.1. Заголовочный файл
Файл: `./mpi/ops_mpi.hpp`.
```cpp
#pragma once

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "task/include/task.hpp"

namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit OtcheskovSElemVecAvgMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace otcheskov_s_elem_vec_avg
```

#### 10.3.2. Файл реализации
Файл: `./mpi/ops_mpi.cpp`.
```cpp
#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"

namespace otcheskov_s_elem_vec_avg {

OtcheskovSElemVecAvgMPI::OtcheskovSElemVecAvgMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = NAN;
}

bool OtcheskovSElemVecAvgMPI::ValidationImpl() {
  return (!GetInput().empty() && std::isnan(GetOutput()));
}

bool OtcheskovSElemVecAvgMPI::PreProcessingImpl() {
  return true;
}

bool OtcheskovSElemVecAvgMPI::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int proc_rank{};
  int proc_num{};
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

  const size_t total_size = GetInput().size();
  const size_t batch_size = total_size / proc_num;
  const size_t proc_size = batch_size + (proc_rank == proc_num - 1 ? total_size % proc_num : 0);
  auto start_local_data = GetInput().begin() + static_cast<std::vector<int>::difference_type>(proc_rank * batch_size);
  auto end_local_data = start_local_data + static_cast<std::vector<int>::difference_type>(proc_size);

  int local_sum = std::reduce(start_local_data, end_local_data);
  int total_sum = 0;
  MPI_Allreduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = total_sum / static_cast<double>(total_size);
  return !std::isnan(GetOutput());
}

bool OtcheskovSElemVecAvgMPI::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg
```

### 10.4. Приложение №4 — функциональные и валидационные тесты
Файл: `./tests/functional/main.cpp`.
```cpp
#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_elem_vec_avg {
// функциональные тесты
class OtcheskovSElemVecAvgFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string filename = FormatFileName(std::get<0>(test_param));
    std::string avg_str = FormatAverage(std::get<1>(test_param));
    return filename + "_" + avg_str;
  }

 protected:
  // открывает файл и считывает элементы вектора
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string filename = std::get<0>(params);
    expected_avg_ = std::get<1>(params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_otcheskov_s_elem_vec_avg, filename);
    std::ifstream file(abs_path);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }

    int num{};
    while (file >> num) {
      input_data_.push_back(num);
    }
    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::fabs(expected_avg_ - output_data) < std::numeric_limits<double>::epsilon();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_avg_ = NAN;
  // форматирует название файла для имени теста
  static std::string FormatFileName(const std::string &filename) {
    size_t lastindex = filename.find_last_of('.');
    std::string name = filename;
    if (lastindex != std::string::npos) {
      name = filename.substr(0, lastindex);
    }

    std::string format_name = name;
    for (char &c : format_name) {
      if (std::isalnum(c) == 0 && c != '_') {
        c = '_';
      }
    }
    return format_name;
  }
  // форматирует ожидаемое выходное значение для имени теста
  static std::string FormatAverage(double value) {
    std::string str = RemoveTrailingZeros(value);
    if (value < 0) {
      str = "minus_" + str.substr(1, str.size());
    }

    for (char &c : str) {
      if (c == '.') {
        c = 'p';
      }
    }
    return "num_" + str;
  }
  // Убирает плавающие 0 в ожидаемом выходном значении
  static std::string RemoveTrailingZeros(double value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(10) << value;

    std::string str_value = ss.str();
    if (str_value.find('.') != std::string::npos) {
      str_value = str_value.substr(0, str_value.find_last_not_of('0') + 1);
      if (str_value.find('.') == str_value.size() - 1) {
        str_value = str_value.substr(0, str_value.size() - 1);
      }
    }
    return str_value;
  }
};

namespace {

TEST_P(OtcheskovSElemVecAvgFuncTests, VectorAverageFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {std::make_tuple("test_vec1.txt", 50.5),
                                            std::make_tuple("test_vec2.txt", 14.5),
                                            std::make_tuple("test_vec_one_elem.txt", 5.0),
                                            std::make_tuple("test_vec_fraction.txt", 4.0 / 3.0),
                                            std::make_tuple("test_vec_one_million_elems.txt", -2.60988),
                                            std::make_tuple("test_vec_alternating_elems.txt", 0.0),
                                            std::make_tuple("test_vec_zeros_elems.txt", 0.0)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OtcheskovSElemVecAvgMPI, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg),
    ppc::util::AddFuncTask<OtcheskovSElemVecAvgSEQ, InType>(kTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = OtcheskovSElemVecAvgFuncTests::PrintFuncTestName<OtcheskovSElemVecAvgFuncTests>;

INSTANTIATE_TEST_SUITE_P(VectorAverageFuncTests, OtcheskovSElemVecAvgFuncTests, kGtestValues, kFuncTestName);


// Валидацонные тесты
class OtcheskovSElemVecAvgFuncTestsValidation : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    if (std::isnan(output_data)) {
      return true;
    }
    return std::fabs(expected_avg_ - output_data) < std::numeric_limits<double>::epsilon();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
  // переопределяем функцию запуска из ppc::util::BaseRunFuncTests
  void ExecuteTest(::ppc::util::FuncTestParam<InType, OutType, TestType> test_param) {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(::ppc::util::GTestParamIndex::kNameTest)>(test_param);

    ValidateTestName(test_name);

    const auto test_env_scope = ppc::util::test::MakePerTestEnvForCurrentGTest(test_name);

    if (IsTestDisabled(test_name)) {
      GTEST_SKIP();
    }

    if (ShouldSkipNonMpiTask(test_name)) {
      std::cerr << "kALL and kMPI tasks are not under mpirun\n";
      GTEST_SKIP();
    }

    task_ =
        std::get<static_cast<std::size_t>(::ppc::util::GTestParamIndex::kTaskGetter)>(test_param)(GetTestInputData());
    const TestType &params = std::get<static_cast<std::size_t>(::ppc::util::GTestParamIndex::kTestParams)>(test_param);
    const std::string param_name = std::get<0>(params);
    // Специально для теста изменяем состояние входных и выходных данных до запуска теста
    if (param_name.find("_changed_output_") != std::string::npos) {
      task_->GetInput() = {1, 1, 1, 1, 1};
      task_->GetOutput() = 1.0;
    }
    ExecuteTaskPipeline();
  }
  // Переопределяем функцию запуска пайплайна из ppc::util::BaseRunFuncTests
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void ExecuteTaskPipeline() {
    EXPECT_FALSE(task_->Validation()); // проверяем провал валидации
    task_->PreProcessing();            // активируем остальные шаги для корректного удаления задачи
    task_->Run();
    task_->PostProcessing();
  }

 private:
  InType input_data_;
  OutType expected_avg_ = NAN;
  ppc::task::TaskPtr<InType, OutType> task_;
};

TEST_P(OtcheskovSElemVecAvgFuncTestsValidation, VectorAverageFuncTestsValidation) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 2> kValidationTestParam = {std::make_tuple("test_empty_vec", NAN),
                                                      std::make_tuple("test_changed_output_before_run", NAN)};

const auto kValidationTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSElemVecAvgMPI, InType>(
                                                         kValidationTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg),
                                                     ppc::util::AddFuncTask<OtcheskovSElemVecAvgSEQ, InType>(
                                                         kValidationTestParam, PPC_SETTINGS_otcheskov_s_elem_vec_avg));

const auto kValidationGtestValues = ppc::util::ExpandToValues(kValidationTestTasksList);

const auto kValidationFuncTestName =
    OtcheskovSElemVecAvgFuncTestsValidation::PrintFuncTestName<OtcheskovSElemVecAvgFuncTestsValidation>;

INSTANTIATE_TEST_SUITE_P(VectorAverageFuncTestsValidation, OtcheskovSElemVecAvgFuncTestsValidation,
                         kValidationGtestValues, kValidationFuncTestName);

}  // namespace

}  // namespace otcheskov_s_elem_vec_avg
```

### 10.5. Приложение №5 — проиводительные тесты
Файл: `./tests/performance/main.cpp`.
```cpp
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_avg_ = NAN;

  void SetUp() override {
    std::string filename = "test_vec_one_million_elems.txt";
    expected_avg_ = -2.60988;

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_otcheskov_s_elem_vec_avg, filename);
    std::ifstream file(abs_path);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }

    int num{};
    while (file >> num) {
      input_data_.push_back(num);
    }
    file.close();

    // 2 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 4 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 8 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 16 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 32 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 64 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 128 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 256 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 512 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::fabs(expected_avg_ - output_data) < std::numeric_limits<double>::epsilon();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(OtcheskovSElemVecAvgPerfTests, VectorAveragePerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, OtcheskovSElemVecAvgMPI, OtcheskovSElemVecAvgSEQ>(
    PPC_SETTINGS_otcheskov_s_elem_vec_avg);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OtcheskovSElemVecAvgPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(VectorAveragePerfTests, OtcheskovSElemVecAvgPerfTests, kGtestValues, kPerfTestName);

}  // namespace otcheskov_s_elem_vec_avg

```