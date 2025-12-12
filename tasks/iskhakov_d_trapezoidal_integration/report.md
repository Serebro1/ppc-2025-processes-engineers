# Интегрирование – метод трапеций

**Студент:** Исхаков Дамир Айратович, группа 3823Б1ПР5
**Технологии:** SEQ-MPI. 
**Вариант:** '20'

## 1. Введение
Реализация последовательного и параллельного вычисления интеграла методом трапеции и сравнение этих реализаций относительно производительности


## 2. Постановка задачи
**Формальная задача**: Численное интегрирование методом трапеций
- Вычисление определенного интеграла ∫[lower_level, top_level] f(x)dx ≈ step/2 * [f(lower_level) + 2∑f(x_i) + f(top_level)]
- Где step = (top_level-lower_level)/number_steps
- x_i = lower_level + i*step

**Входные данные**:
- Нижний предел (lower_level), верхний предел (top_level), количество шагов (number_steps), функция f(x)
- Структура данных: `std::tuple<double, double, std::function<double(double)>, int>`

**Выходные данные**:
- Приближенное значение определенного интеграла (double)

**Ограничения**:
- lower_level < top_level (проверка корректности пределов интегрирования)
- number_steps > 0 (положительное количество шагов)
- Функция должна быть определена на [lower_level, top_level]

## 3. Базовый алгоритм (Последовательный)

1. Проверка входных данных: убедиться, что lower_level < top_level и number_steps > 0
2. Вычисление шага: Где step = (top_level-lower_level)/number_steps
3. Вычисление вклада концевых точек: (f(lower_level) + f(top_level))/2
4. Суммирование внутренних точек: ∑f(lower_level + step*top_level)
5. Умножение на шаг

(Псевдокод)

if (lower_level < top_level) && (number_steps > 0); {return true};
double step = (top_level - lower_level) / number_steps;

result = (input_function(lower_level) + input_function(top_level)) / 2.0;

for (int i = 1; i < number_steps; i++) {
result += input_function(lower_level + step * i);
}

result *= step;

## 4. Схема распараллеливания
- каждый процесс, получает свои точки, через блочное распределение через MPI_Scatterv

и так далее в зависимости от числа процессов

- Цикл вычисления всех точек для каждого процесса происходит в процессе ранга 0, после чего рассылается по другим процессам

MPI_Scatterv(points.data(), elements_per_proc.data(), displacement.data(), 
             MPI_DOUBLE, local_points.data(), local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// Локальные вычисления
double local_sum = 0.0;
for (double point : local_points) {
    local_sum += input_function(point);
}

// Коррекция граничных точек для метода трапеций
if (world_rank == 0) {
    local_sum -= input_function(local_points[0]) * 0.5;
}
if (world_rank == world_size - 1) {
    local_sum -= input_function(local_points.back()) * 0.5;
}

// Глобальное суммирование
MPI_Allreduce(&local_sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

- Процесс 0 ранга получает задачу и раcсылает данные другим процессам, после чего все процессы вносят свои результаты работы в окончательный результат
    MPI_Allreduce(&local_sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


## 5. Детали реализации
- Весь код был выполнен в шаблоне задачи, с добавлением 1 функции, Function, необходимой для лучшей читаемости кода

## 6. Экспериментальная установка
- **Hardware/OS:** 
 * Процессор: 12th Gen Intel(R) Core(TM) i5-12500Hl
 * Ядра/Потоки: 12 ядер, 16 потоков
 * ОЗУ: 8 GB DDR 4
 * ОС: Linux Mint 22.2

- **Toolchain:** 
 * gcc --version 13.3.0
 * mpirun (Open MPI) 4.1.6
 * cmake version 3.28.3

- **Environment:** 
 * Количество процессов MPI

- **Data:**  
 * Функция: f(x) = x³·sin(x) + 2·cos(x)
 * Интервалы интегрирования: [0,1], [0,2], [1,3]
 * Количество шагов: 10K-300K для различных тестовых случаев
 * Эталонные значения вычислены аналитически

## 7. Результаты и обсуждение

### 7.1 Корректность
**Метод проверки**:
- Сравнение с известными аналитическими решениями
- Порог относительной ошибки: < 1%
- Несколько тестовых случаев с разными интервалами

**Тестовые случаи**:
- ∫[0,1] f(x)dx ≈ 1.8600 (проверено)
- ∫[0,2] f(x)dx ≈ 5.6100 (проверено)  
- ∫[1,3] f(x)dx ≈ 10.2953 (проверено)

**Проверка инвариантов**:
- Результат не зависит от количества процессов (MPI)
- Сохранение: удвоение количества шагов уменьшает ошибку вдвое (сходимость O(h²))

### 7.2 Производительность
Present time, speedup and efficiency. Example table:

| Mode  | Count    | Time, s   | Speedup   | Efficiency    |
|-------|----------|-----------|-----------|---------------|
| seq   | 1        | 0.07079   | 1.00      | N/A           |
| mpi   | 2        | 0.10681   | 0.66      | 33.1%         |
| mpi   | 4        | 0.09319   | 0.76      | 19.0%         |

- Программа показала лишь замедление выполнения (0.76× на 4 процессах)
- Низкая эффективность (<20% на 4 процессах)

## 8. Выводы
Реализация MPI не обеспечило значительного ускорения

## 9. Ссылки
1. Документация по курсу - https://learning-process.github.io/parallel_programming_course/ru/common_information/report.html
2. Записи лекций - https://disk.yandex.ru/d/NvHFyhOJCQU65w
3. Гугл - https://www.google.com/

