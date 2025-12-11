# Average Vector Mean (Gaivoronskiy M.)

- Student: Gaivoronskiy Maksim, group 3823Б1ПР1
- Technology: SEQ, MPI
- Task number: 2

## 1. Introduction
Цель работы — реализовать задачу вычисления среднего значения элементов вектора двумя способами: последовательно и с использованием MPI. В отчёте описаны алгоритмы, схема распараллеливания и измерения производительности, достаточные для воспроизведения результатов.

## 2. Problem Statement
На вход поступает вектор вещественных чисел фиксированной длины *N*. Требуется вернуть среднее арифметическое:  
`avg = (1 / N) * Σ_i v[i]`. Ограничения: `N > 0`, значения хранятся в `std::vector<double>`, результат — `double`.

## 3. Baseline Algorithm (Sequential)
Базовый алгоритм выполняет один проход по вектору, аккумулирует сумму в `double` (через `std::accumulate`) и делит её на `N`. Важные детали:
- валидация проверяет непустой вход;
- данные копируются на этапе `PreProcessing` для единообразия с MPI‑версией;
- на этапе `PostProcessing` рассчитывается окончательное среднее и проверяется `std::isfinite`.

## 4. Parallelization Scheme
MPI‑версия использует одноразовый Scatter/Reduce‑паттерн:
- **Распределение данных**: корневой процесс (ранг 0) делит исходный вектор на почти равные блоки (разница не превышает 1 элемент) и рассылает их через `MPI_Scatterv`.
- **Локальные вычисления**: каждый ранг суммирует свой блок.
- **Глобальное объединение**: `MPI_Allreduce` собирает локальные суммы и возвращает итоговую сумму на все ранги.
- **Финализация**: корень (и все ранги) делят глобальную сумму на `N` и записывают результат.
Топология — полносвязная через `MPI_COMM_WORLD`, дополнительных синхронизаций нет.

## 5. Implementation Details
- Код расположен в `seq/src/ops_seq.cpp` и `mpi/src/ops_mpi.cpp`, интерфейсы — в `seq/include` и `mpi/include`.
- Типы описаны в `common/include/common.hpp`: вход — `std::vector<double>`, выход — `double`, тесты оперируют структурами `{file_name, expected_average}`.
- Функциональные тесты читают входы из текстовых файлов (`data/test_vec_*.txt`), что позволяет легко расширять покрытие и повторять результаты.
- MPI‑класс хранит ранг/размер коммуникатора, входной вектор только у ранга 0, распределение осуществляется через `MPI_Scatterv`, суммирование — через `MPI_Allreduce`.
- Perf‑тест строит большие векторы путём повторения блока `perf_vec_base.txt`, поэтому вход идентичен для SEQ и MPI.

## 6. Experimental Setup
- **Hardware / OS**: Apple M1 (8 ядер, 4P+4E), 16 GB LPDDR4X, macOS 14.5 (Darwin 23.5.0).
- **Toolchain**: CMake + Apple Clang 15.0.0, `CMAKE_BUILD_TYPE=Release`.
- **Environment**:
  - Функциональные тесты (SEQ) — запуск `./build/bin/ppc_func_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum*'`.
  - MPI тесты — `mpirun -np 4 ./build/bin/ppc_func_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum_mpi*'`.
  - Perf — `ppc_perf_tests` и под `mpirun -np 4`, дополнительно прогоны с `-np 8`.
- **Данные**:
  - Функциональные кейсы: файлы `test_vec_small.txt`, `test_vec_mixed.txt`, `test_vec_single.txt`, `test_vec_progression.txt`, `test_vec_fraction.txt`, `test_empty_vec.txt`.
  - Perf: базовый блок `perf_vec_base.txt` (100 чисел), дублируемый до 1 000 000 / 5 000 000 / 20 000 000 / 100 000 000 элементов.
- **Команды**:
  ```bash
  cmake --build build -j4
  ./build/bin/ppc_func_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum*'
  mpirun -np 4 ./build/bin/ppc_func_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum_mpi*'
  ./build/bin/ppc_perf_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum_seq*'
  mpirun -np 4 ./build/bin/ppc_perf_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum_mpi*'
  mpirun -np 8 ./build/bin/ppc_perf_tests '--gtest_filter=*gaivoronskiy_m_average_vector_sum_mpi*'
  ```

## 7. Results and Discussion

### 7.1 Correctness
- Функциональные тесты (`ppc_func_tests`) читают реальные входы из файлов, проверяют совпадение среднего с эталоном (`|Δ| ≤ 1e-9`). Покрываются позитивные случаи и пустой вектор.
- MPI‑мод выполняется под `mpirun -np 4`. Все проверки проходят без расхождений.

### 7.2 Performance
Все значения времени — усреднённые выводы `ppc_perf_tests` (режим `pipeline`). Speedup рассчитывается относительно последовательного `pipeline`‑времени на том же размере данных; эффективность = `speedup / proc_count`.

**Размер = 1 000 000 элементов**

| Mode | Processes | Time, s | Speedup | Efficiency |
|------|-----------|---------|---------|------------|
| seq (pipeline) | 1 | 0.00141 | 1.00 | N/A |
| mpi (pipeline) | 4 | 0.00190 | 0.74 | 18% |
| mpi (pipeline) | 8 | 0.00414 | 0.34 | 4% |

**Размер = 5 000 000 элементов**

| Mode | Processes | Time, s | Speedup | Efficiency |
|------|-----------|---------|---------|------------|
| seq (pipeline) | 1 | 0.00787 | 1.00 | N/A |
| mpi (pipeline) | 4 | 0.00702 | 1.12 | 28% |
| mpi (pipeline) | 8 | 0.01195 | 0.66 | 8% |

**Размер = 20 000 000 элементов**

| Mode | Processes | Time, s | Speedup | Efficiency |
|------|-----------|---------|---------|------------|
| seq (pipeline) | 1 | 0.03445 | 1.00 | N/A |
| mpi (pipeline) | 4 | 0.03158 | 1.09 | 27% |
| mpi (pipeline) | 8 | 0.02670 | 1.29 | 16% |

**Размер = 100 000 000 элементов**

| Mode | Processes | Time, s | Speedup | Efficiency |
|------|-----------|---------|---------|------------|
| seq (pipeline) | 1 | 0.20822 | 1.00 | N/A |
| mpi (pipeline) | 4 | 0.19048 | 1.09 | 27% |
| mpi (pipeline) | 8 | 0.32813 | 0.63 | 8% |

MPI выигрывает на размерах ≥5 млн, но масштабирование выше 4 процессов на одном узле приводит к деградации (узкое место — общая память и сеть).

## 8. Conclusions
- Реализованы SEQ и MPI версии среднего значения, корректность подтверждена файловыми тестами.
- Продуктивность MPI растёт вместе с размером входа: при 100 М элементов 4 процесса дают ~9 % выигрыш, но 8 процессов на одном узле уже проигрывают из‑за конкуренции за ресурсы.
- Для дальнейшего улучшения требуется межузловое масштабирование, оптимизация обменов (`MPI_Reduce`/`Allreduce` с деревьями), более крупные блоки данных и привязка процессов к ядрам (pinning).

## 9. References
1. MPI Forum, *MPI: A Message-Passing Interface Standard Version 4.0*, https://www.mpi-forum.org
2. ISO/IEC 14882:2020, *Programming Languages — C++*.

## Appendix (Optional)
```cpp
// Отрывок вычисления локальной суммы в MPI-варианте
local_sum_ = std::accumulate(local_buffer_.begin(), local_buffer_.end(), 0.0);
MPI_Allreduce(&local_sum_, &global_sum_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
GetOutput() = global_sum_ / static_cast<double>(total_size_);
```

