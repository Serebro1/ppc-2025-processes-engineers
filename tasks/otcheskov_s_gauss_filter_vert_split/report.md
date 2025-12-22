# Линейная фильтрация изображений (вертикальное разбиение). Ядро Гаусса 3x3.

- Студент: Отческов Семён Андреевич, группа 3823Б1ПР1
- Технологии: SEQ | MPI
- Вариант: 27


## 1. Введение
- Линейная фильтрация изображений является одной из основных тем, изучаемых в компьютерном зрении и обработке изображений. Процесс заключается в применении фильтра (ядра свёртки) к изображению. Одним из таких фильтров является ядро Гаусса, применяемый для сглаживания и устранения шума изображения.
- В данной работе представлены два алгоритма линейной фильтрации изображений с использованием ядра Гаусса 3×3: последовательная (базовая) реализация и параллельная реализация с вертикальным разбиением изображения и использованием технологии MPI (Message Passing Interface).

- **Цель работы:** сравнение производительности алгоритмов и анализ эффективности распараллеливания вычислительной задачи при обработке изображений различного размера.


## 2. Постановка задачи
**Формальная постановка:**
- Для каждого пикселя изображения *I* с координатами *(x,y)* и для каждого цветового канала *c* вычислить новое значение путем свертки с гауссовым ядром 3x3: 
  - $I_{out}\left(x, y, c\right)=\sum\limits_{i=-1}^{1}{\sum\limits_{j=-1}^{1}{I_{mirrored}\left(x+i,y+j,c\right) \cdot K\left(i,j\right)}}$
  - $k\left(i, j\right)$ — веса ядра Гаусса
  - $I_{mirrored}$ — изображение с обработанными границами методом симметричного отражения с дублированием крайних пикселей.
- Ядро Гаусса 3x3 имеет следующий вид:
  - $K=\frac{1}{16}\ast\begin{bmatrix}
      1 & 2 & 1  \\
      2 & 4 & 2 \\
      1 & 2 & 1 
      \end{bmatrix}$

**Входные данные:**
- Изображение в виде трехмерного массива (высота × ширина × количество каналов).
- Значения пикселей представлены целыми числами в диапазоне [0, 255].
- Поддерживаются изображения с различным количеством цветовых каналов (1 для градаций серого, 3 для RGB и т.д.).

**Выходные данные:**
- Отфильтрованное изображение того же размера и формата, что и входное.

**Ограничения:**
- Количество цветовых каналов должно быть положительным целым числом.


## 3. Описание алгоритма (последовательного)

### 3.1. Этапы выполнения задачи
**1. Валидация данных (`ValidationImpl`):**
- Проверка на пустоту входного изображения.
- Проверка корректности объёма данных (ширина, высота, число каналов должны быть больше нуля).
- Проверка соответствия объема данных заявленным размерам изображения.

**2. Предобработка данных (`PreProcessingImpl`):**
- Задача не требует предобработки, поэтому данный этап пропускается.

**3. Вычисления (`RunImpl`):**
- Выделение памяти для выходного изображения с сохранением размеров входного.
- Последовательный обход всех пикселей изображения по строкам и столбцам:
  - Для каждого пикселя и каждого цветового канала:
    - Инициализация суммы взвешенных значений.
    - Перебор всех 9 элементов ядра Гаусса (окно $3×3$).
    - Для каждой позиции ядра вычисление координат соответствующего пикселя с учетом обработки границ `методом отражения с дублированием краёв`.
    - Умножение значения пикселя на соответствующий вес из ядра Гаусса и добавление к сумме.
  - Округление итоговой суммы и ограничение диапазона значений [0, 255].
  - Запись результата в выходное изображение.

- Метод отражения:
  - Если индекс выходит за верхнюю границу ($y<0$), то $y=-y-1$.
  - Если индекс выходит за нижнюю границу ($y\geqslant height$), то $y=2*height-y-1$.
  - Если индекс выходит за левую границу ($x<0$), то $x=-x-1$.
  - Если индекс выходит за правую границу ($x\geqslant width$), то $x=2*width-x-1$.


**4. Постобработка данных (`PostProcessingImpl`):**
- Задача не требует постобработки, поэтому данный этап пропускается.

### 3.2. Сложность алгоритма:
- Временная сложность: $O(H \cdot W \cdot C)$ с константным множителем 9
  - $H$ — высота изображения.
  - $W$ — ширина изображения.
  - $C$ — количество каналов изображения.
  - Для каждого пикселя выполняется фиксированное число операций — 9 умножений и сложений в случае ядра Гаусса $3×3$.
- Пространственная сложность: $O(H \cdot W \cdot C)$ — хранение входного и выходного изображений.

## 4. Схема распараллеливания алгоритма с помощью MPI
Реализация данной схемы представлена в [Приложении №3 — параллельная версия решения задачи](#103-приложение-3--параллельная-версия-решения-задачи).
### 4.1. Распределение данных
- Вертикальное разбиение изображения:
  - Процесс ранга 0 распределяет данные между всеми процессами с помощью `MPI_Scatterv`
  - Размер количество столбцов (`base_cols`) определяется как `ширина изображения / число активных процессов`.
  - Если ширина изображения меньше числа процессов, используются только `active_procs_ = min(proc_num_, width)` процессов.
  - Если есть остаток после деления числа столбцов, то он добавляется по одному столбцу первым `remainder` процессам.
  - Каждый процесс получает вертикальную полосу изображения в локальный буфер `local_data`.
  - Размер локальной полосы для процесса: `local_width_ = base_cols + (proc_rank_ < remainder ? 1 : 0)`.

### 4.2. Топология коммуникаций
- Линейная топология с соседними связями:
  - все процессы связаны через `MPI_COMM_WORLD`.
  - Дополнительные связи формируются динамически для обмена граничными столбцами.
- Роль процессов:
  - **Ранг 0**: координатор — вычисляет распределение, рассылает метаданные, распределяет данные, собирает результаты, обрабатывает левую границу изображения.
  - **Ранг N-1 (active_procs_ - 1)**: обрабатывает правую границу изображения.
  - **Промежуточные ранги**: Обмениваются данными с левыми и правыми соседями для формирования расширенной области, что способствует уменьшению обращений к процессам при фильтрации границ своих локальных частей.

### 4.3. Паттерны коммуникации
- Функция `MPI_Bcast`:
  - Применяется в валидации, чтобы каждый процесс прошёл проверку аналогично процессу ранга 0.
  - Применяется для рассылки метаданных об изображении.
- Функция `MPI_Scatterv`:
  - Распределение вертикальных полос изображения с учётом неравномерного деления.
  - Каждый процесс получает свою часть данных согласно своему числу обрабатываемых элементов и смещению в глобальном массиве.
- Функция `MPI_Sendrecv`:
  1. Двусторонний обмен граничными столбцами между соседними процессами.
  2. Каждый процесс одновременно отправляет левый столбец левому соседу и правый столбец правому соседу.
  3. Используются разные теги сообщений (0 для левого обмена, 1 для правого) для избежания конфликтов.
- Функция `MPI_Gatherv`:
  - Асимметричный сбор обработанных полос на процессе 0.
  - Учитывает неравномерное распределение столбцов при формировании итогового изображения.

### 4.4. Распределение вычислений
1. **Рассылка метаданных:**
  - Процесс 0 рассылает метаданные изображения всем процессам.
2. **Вертикальное разбиение:**
  - Процесс 0 вычисляет распределение столбцов и формирует буфер для `MPI_Scatterv`.
  - Каждый процесс получает свою вертикальную полосу изображения.
3. **Обмен граничными столбцами:**
  - Формирование локальных граничных столбцов для обмена.
  - Двусторонний обмен с соседними процессами.
  - Создание расширенного локального изображения (`extended_data_`) размером (`local_width_ + 2`).
    - Левые и правые границы, которые находятся у первого и последнего процессов соответственно, дублируются.
    - В остальном случае заносим полученные границы соседей и локальные в расширенные данные.
  - После создания расширенной области можно освободить локальный входной буфер (`local_data_`).
4. **Локальная фильтрация:**
  - Каждый процесс применяет гауссов фильтр 3×3 к своей полосе.
  - Для пикселей у внутренних границ используются границы из расширенных данных.
  - Верхние и нижние границы обрабатываются методом отражения.
  - Результат сохраняется в локальный выходной буфер (`local_output_`).
5. **Сбор результатов:**
  - Процесс 0 собирает обработанные полосы (`local_output_`).
  - Формирование итогового изображения построчным копированием блоков данных из расширенной версии локальных полос.
  - Неактивные процессы (`proc_rank_` ≥ `active_procs_`) участвуют в коммуникации с нулевыми размерами.


## 5. Особенности реализаций

### 5.1. Структура кода
Реализации классов и методов на языке С++ указаны в [Приложении](#10-приложение).
```
tasks/otcheskov_s_gauss_filter_ver_split/
├── common
│ └── include
│     └── common.hpp
├── data
│   ├── chess.jpg
│   └── gradient.jpg
├── mpi
│   ├── include
│   │   └── ops_mpi.hpp
│   └── src
│       └── ops_mpi.cpp
├── seq
│   ├── include
│   │   └── ops_seq.hpp
│   └── src
│       └── ops_seq.cpp
├── tests
│   ├── functional
│   │   └── functional.cpp
│   └── performance
│       └── performance.cpp
├── report.md
├── settings.json
└── info.json
    
```
#### 5.1.1. Файлы
- `./common/include/common.hpp` — общие определения типов данных ([см. Приложение №1](#101-приложение-1--общие-определения)).
- `./seq/include/ops_seq.hpp` — определение класса последовательной версии задачи ([см. Приложение №2.1](#1021-заголовочный-файл)).
- `./mpi/include/ops_mpi.hpp` — определение класса параллельной версии задачи ([см. Приложение №2.2](#1021-файл-реализации)).
- `./seq/src/ops_seq.cpp` — реализация последовательной версии задачи ([см. Приложение №3.1](#1031-заголовочный-файл)).
- `./mpi/src/ops_mpi.cpp` — реализация параллельной версии задачи ([см. Приложение №3.2](#1032-файл-реализации)).
- `./tests/functional/main.cpp` — реализация функциональных и валидационных тестов ([см. Приложение №4](#104-приложение-4--функциональные-и-валидационные-тесты)).
- `./tests/performance/main.cpp` — реализация производительных тестов ([см. Приложение №5](#105-приложение-5--проиводительные-тесты)).

#### 5.1.2. Ключевые классы
- `OtcheskovSGaussFilterVertSplitSEQ` — последовательная версия.
- `OtcheskovSGaussFilterVertSplitMPI` — параллельная версия.
- `OtcheskovSGaussFilterVertSplitValidationTests` — валидационные тесты.
- `OtcheskovSGaussFilterVertSplitFuncTests` — функциональные тесты на сгенерированных данных.
- `OtcheskovSGaussFilterVertSplitRealTests` — тесты с изображениями.
- `OtcheskovSGaussFilterVertSplitPerfTests` — производительные тесты.

#### 5.1.3. Основные методы
- `ValidationImpl` — валидация входных данных и состояния выходных данных.
- `PreProcessingImpl` — препроцессинг, не используется.
- `RunImpl` — основная логика вычислений.
- `PostProcessingImpl` — постпроцессинг, не используется.

### 5.2. Реализация последовательной версии

#### 5.2.1. ValidationImpl
- Проверка непустоты массива пикселей `data`.
- Проверка минимальных размеров изображения (не менее $3×3$ пикселей).
- Проверка положительного количества цветовых каналов.
- Соответствие объёма данных `data` с заявленными в метаданных размерами (`data.size() == height * width * channels`).
- Результат проверки кэшируется в поле `is_valid_` для последующего использования в `RunImpl`, для корректной работы тестов.

**Реализация на C++:**
```c++
bool OtcheskovSGaussFilterVertSplitSEQ::ValidationImpl() {
  const auto &[metadata, data] = GetInput();
  is_valid_ = !data.empty() && 
             (metadata.height > 0 && metadata.width > 0 && metadata.channels > 0) &&
             (data.size() == metadata.height * metadata.width * metadata.channels);
  return is_valid_;
}
```

#### 5.2.2. PreProcessingImpl
В препроцессинге нет необходимости, поэтому данный этап пропускается.

#### 5.2.3. RunImpl
1. Проверка валидности данных:
  - Используется закэшированное значение `is_valid` из этапа валидации.
2. Подготовка выходного буфера:
  - Создаётся массив пикселей того же размера, что и входной.
3. Обработка каждого пикселя:
  - Для каждого пикселя `(row, col)` и каждого цветового канала `ch`:
    - Инициализация суммы взвешенных значений.
    - Применяется окно $3×3$ с использованием ядра Гаусса функцией `process_pixel`.
    - Обработка границ через отражение координат лямбда-функцией `mirror_coord`:
      - Для отрицательных индексов: `pos = -pos - 1`.
      - Для индексов за границей: `pos = 2*size - pos - 1`.
    - Результат нормализуется в диапазон [0, 255] с использованием `std::clamp`.
4. Запись результата:
  - Вычисленное значение сохраняется в выходной массив по индексу, рассчитанному как `((row * width + col) * channels) + channel`.

#### 5.2.4. Вспомогательные методы в RunImpl

**mirror_coord:**
- Реализует метод отражения для обработки границ:
```cpp
auto mirror_coord = [&](const size_t &current, const int &off, const size_t &size) -> size_t {
    int64_t pos = static_cast<int64_t>(current) + off;
    if (pos < 0) {
      return static_cast<size_t>(-pos - 1);
    }
    if (std::cmp_greater_equal(static_cast<size_t>(pos), size)) {
      return (2 * size) - static_cast<size_t>(pos) - 1;
    }
    return static_cast<size_t>(pos);
  };
```
**ProcessPixel:**
- Реализует метод обработки пикселя ядром Гаусса
```cpp
uint8_t OtcheskovSGaussFilterVertSplitSEQ::ProcessPixel(size_t row, size_t col, size_t ch) {
  const auto &[in_meta, in_data] = GetInput();
  const auto &[height, width, channels] = in_meta;

  auto mirror_coord = [&](const size_t &current, const int &off, const size_t &size) -> size_t {
    int64_t pos = static_cast<int64_t>(current) + off;
    if (pos < 0) {
      return static_cast<size_t>(-pos - 1);
    }
    if (static_cast<size_t>(pos) >= size) {
      return (2 * size) - static_cast<size_t>(pos) - 1;
    }
    return static_cast<size_t>(pos);
  };

  double sum = 0.0;
  for (int dy = 0; dy < 3; ++dy) {
    size_t src_y = mirror_coord(row, dy - 1, height);
    for (int dx = 0; dx < 3; ++dx) {
      size_t src_x = mirror_coord(col, dx - 1, width);
      double weight = kGaussianKernel.at(dy).at(dx);
      size_t src_idx = (((src_y * width) + src_x) * channels) + ch;
      sum += weight * in_data[src_idx];
    }
  }
  return static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
}
```

#### 5.2.5. PostProcessingImpl
В постпроцессинге нет необходимости, поэтому данный этап пропускается.

### 5.3. Реализация параллельной версии
**Этапы:** `PreProcessingImpl`, `PostProcessingImpl` аналогичны последовательной версии.

#### 5.3.1. Конструктор и инициализация
- **Инициализация MPI окружения:**
  - Определяется ранг текущего процесса и общее количество процессов.
  - Только процесс с рангом 0 загружает входное изображение.

**Реализация на C++:**
```cpp
int proc_rank{};
int proc_num{};
MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
SetTypeOfTask(GetStaticTypeOfTask());

proc_rank_ = static_cast<size_t>(proc_rank);
proc_num_ = static_cast<size_t>(proc_num);
if (proc_rank_ == 0) {
  GetInput() = in;
}
```
#### 5.3.2. ValidationImpl
- Выполняется проверка на процессе ранга 0, аналогична реализации в последовательной версии.
- Результат проверки рассылается всем процессам через `MPI_Bcast` и кэшируется в переменную `is_valid_`.
- Таким образом, проверка едина для всех процессов.

#### 5.3.3. RunImpl
- Дополнительная валидация по переменной `is_valid_`.
- Рассылка метаданных всем процессам.
- Распределение данных, как описано в разделе [4.1. Распределение данных](#41-распределение-данных).
- Обмен граничными столбцами между соседними процессами, как описано в разделе [4.4. Распределение вычислений](#44-распределение-вычислений) в пункте 3.
- Фильтрация локальной полосы изображения:
  - Каждый активный процесс обрабатывает свою полученную полосу, неактивные ничего не выполняют.
  - Верхняя и нижняя границы обрабатываются отражением с дублированием крайних пикселей.
  - Левые и правые границы известны и находятся в расширенных данных (`extended_data_`).
- Сбор результатов:
  - Применение `MPI_Gatherv` для асиметричного сбора данных.
  - Формирование итогового изображения на процессе ранга 0 через построчное копирование:
    ```cpp
    for (size_t proc = 0; proc < active_procs_; ++proc) {
        const size_t cols = base_cols + (proc < remain ? 1 : 0);
        const size_t start_col = (base_cols * proc) + std::min(proc, remain);
        const uint8_t *src = recv_buffer.data() + static_cast<size_t>(buffer_offset) + (i * cols * channels);
        uint8_t *dst = out_data.data() + (i * row_size) + (start_col * channels);
        std::memcpy(dst, src, cols * channels);

        buffer_offset += counts[proc];
      }
    ```

#### 5.3.4. Основные методы (особенности реализации)
**DistributeData**
 - Оптимизация для избыточных процессов — неактивные получают пустые данные.
 - Эффективное формирование буфера:
   - Предварительное копирование данных в `send_buffer` с сохранением непрерывности строк:
    ```cpp
    for (size_t row = 0; row < height; ++row) {
      std::memcpy(buf_ptr, src_row, cols * channels_);
      buf_ptr += cols * channels_;
    }
    ```
   - Корректная работа с остатком:
     - Использование `std::min(proc, remain)` для определения смещения первого столбца процесса.

**ExchangeBoundaryColumns**
- Для крайних активных процессов используются отраженные локальные данные вместо обмена.
- Формирование расширенной области:
  - Размер расширенного изображения — `(local_width_ + 2) × height`, что позволяет обрабатывать пиксели на границах блока без дополнительных проверок.
- Использование разных тегов сообщений (0 для левого соседа, 1 для правого) предотвращает смешение данных при их отправлении/получении.

**ApplyGaussianFilter**
- Каждый процесс обрабатывает только свои столбцы, используя расширенные данные.
- Для обработки верхних/нижних границ используется та же логика, что и в последовательной версии.
- Проверка выхода за границы расширенного буфера.

**CollectResults**
- Асиметричный сбор данных.
- Последовательное копирование данных в выходной буфер без доп. транспозиций.

### 5.4. Использование памяти
- **Последовательная версия:**
  - `O(H × W × C)` — хранение полного входного и выходного изображений, где `H` — высота, `W` — ширина, `C` — количество каналов.

- **Параллельная версия:**
  - **Процесс ранга 0:** `O(H × W × C)` (исходное и выходное изображения) + `O(H × (W/P + 2) × C)` (расширенные локальные данные).
  - **Активные процессы**: `O(H × (W/P + 2) × C)` — локальная полоса + два граничных столбца для обмена.
  - **Неактивные процессы**: `O(1)` — минимальные константы для участие в MPI-операциях.

### 5.5. Допущения и крайние случаи
- Все процессы запускаются в рамках одного MPI-коммуникатора.
- Ранг 0 является корневым процессом для распределения данных и валидации.
- Если ширина изображения меньше числа процессов, используются только active_procs_ = min(proc_num_, width) процессов.
- Обработка границ:
  - Для внутренних границ (между процессами) используется корректный обмен данными.
  - Для внешних границ (крайние столбцы всего изображения) применяется дублирование краевых пикселей в соответствии с выбранным методом отражения.
- Пустые процессы: Процессы с `proc_rank ≥ active_procs_` получают пустые задания, но участвуют в коллективных операциях.


## 6. Тестовые инфраструктуры
### 6.1 WSL
| Параметр   | Значение                                             |
| ---------- | ---------------------------------------------------- |
| CPU        | Intel Core i5 12400F (6 cores, 12 threads, 2500 MHz) |
| RAM        | 16 GB DDR4 (3200 MHz)                                |
| OS         | Ubuntu 24.04.3 LTS on Windows 10 x86_64              |
| Компилятор | GCC 13.3.0, Release Build                            |

### 6.2. Общие настройки
- **Переменные окружения:** PPC_NUM_PROC = 2, 4.
- **Данные:** генерируются программно, пара фотографий в директории `otcheskov_s_gauss_filter_vert_split/data`.


## 7. Результаты и обсуждение

### 7.1. Корректность
- Корректность реализации для обоих версий (SEQ и MPI) проверена через три типа тестов: валидационные, функциональные и реальные.
- Все тесты параметризированы и выполняются автоматически через Google Test.

#### 7.1.1. Валидационные тесты
Проверяют обработку некорректных входных данных:
- **Пустые данные:** изображение без пикселей.
- **Несоответствие объёма данных:** фактический размер переданных данных несовпадает с заявленным в метаданных.
- **Некорректные метаданные:** высота/ширина/каналы равны нулю или отрицательны.

#### 7.1.2. Функциональные тесты
Проверяют корректность фильтрации на сгенерированных изображениях:
- **Очень малые изображения:** 1x15x1 пикселей.
- **Малые изображения:** 3×3×1, 3×3×3, 4×4×1 пикселей.
- **Средние изображения:** 10×20×3 пикселей.
- **Тесты границ:** специальные изображения 9×9 и 10×10 для проверки обработки краев методом отражения.
- **Тесты с резкими переходами:** изображение с вертикальными линиями 15×15×3 для проверки подавления артефактов.

**Методика проверки:**
- Для каждого теста генерируется эталонный результат через функцию `ApplyGaussianFilter`.
- Сравнение результатов выполняется поэлементно.
- Для MPI версии результат проверяется только на процессе ранга 0.

#### 7.1.3. Тесты на изображениях
Проверяют корректность фильтрации на изображениях:
- **chess.jpg:** изображение шахматной доски с чёткими границами для проверки сглаживания.
- **gradient.jpg:** резкий градиент для проверки сглаживания.

**Особенности реализации:**
- Изображения загружаются через библиотеку `stb_image` в формате RGB.
- Обработка ошибок загрузки изображений через исключения (`std::runtime_error`).
- Проверка аналогична функциональным тестам.

#### 7.1.4. Механизм проверки
- Все тесты выполняются как для последовательной (SEQ), так и для параллельной (MPI) версии.
- Сравнение результатов: для всех тестов (кроме валидационных) используется точное поэлементное сравнение:
- **MPI-специфика:** для параллельной версии сравнение выполняется только на процессе 0:
  ```cpp
  int proc_rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  if (proc_rank == 0) {
    return expect_img_ == output_data;
  }
  return true;
  ```
  - **Обработка исключений:** ошибки загрузки реальных изображений перехватываются и выводятся в stderr.

- Для некорректных сценариев проверяется провал валидации (`ValidationImpl()`).

### 7.2. Производительные тесты

#### 7.2.1. Методология тестирования
- **Данные:** сгенерированное RGB-изображение размером $10000×10000$ пикселей с резким градиентным заполнением.
- **Режимы:**
  - **pipeline** — запуск и измерение времени всех этапов алгоритма (`Validation -> PreProcessing -> Run -> PostProcessing`).
  - **task_run** — измерение только этапа `Run` (распределение данных + основные вычисления + сбор итогового изображения).
- **Производительность** мерилась только в режиме `task_run`
- **Метрики:**
  - Число процессов.
  - Абсолютное время выполнения.
  - Ускорение: $S_{p}=\frac{T\left(seq\right)}{T\left(p\right)}$
  - Эффективность: $E_{p}=\frac{S_{p}}{p} \ast 100\%$

#### 7.2.2. Результаты тестирования на изображении размером $10000×10000×3$ пикселей

**WSL:**
| Режим | Процессов | Время, s | Ускорение | Эффективность |
| ----- | --------- | -------- | --------- | ------------- |
| seq   | 1         | 2.273410 | 1.0000    | N/A           |
| mpi   | 2         | 1.596012 | 1.4244    | 71.2%         |
| mpi   | 4         | 1.086443 | 2.0925    | 52.3%         |
| mpi   | 6         | 0.849003 | 2.6777    | 44.6%         |
| mpi   | 7\*       | 0.807920 | 2.8139    | 40.2%         |
| mpi   | 8\*       | 0.788059 | 2.8848    | 36.1%         |

**\*\*GitHub:**
| Режим | Процессов | Время, s | Ускорение | Эффективность |
| ----- | --------- | -------- | --------- | ------------- |
| seq   | 1         | 3.165910 | 1.0000    | N/A           |
| mpi   | 2         | 1.817134 | 1.7423    | 87.1%         |

*\*Так как машина имеет 6 физических ядер, использовался ключ --oversubscribe для увеличения числа процессов без ограничений.*
*\*\*Результаты собирались на локальном форке из Github Actions.*

### 7.3. Анализ результатов

- **Эффективность распараллеливания:** Алгоритм успешно распределяет вычислительную нагрузку, что подтверждается ростом ускорения с увеличением числа процессов.
- **Ограничения масштабируемости:**
  - Алгоритм требует значительного обмена данными между процессами для обработки границ, что становится узким местом при большом числе процессов.
  - Для ядра Гаусса 3×3 объем вычислений на пиксель невелик (9 операций умножения-сложения), что усиливает влияние коммуникационных задержек при повышении числа процессов.
- **Ограничения архитектуры:**
  - При использовании большего числа процессов, чем физических ядер (oversubscribe), происходит разделение ресурсов ядра между процессами, что снижает эффективность и ускорение.

## 8. Заключения

### 8.1. Достигнутые результаты:
1. **Корректная реализация параллелизма** — Успешно реализована схема вертикального разбиения изображения с корректной обработкой границ между процессами. Все функциональные тесты подтверждают идентичность результатов SEQ и MPI версий.
2. **Гибкое распределение нагрузки** — алгоритм динамически адаптируется к различному количеству процессов и размерам изображения, включая обработку случаев, когда ширина изображения меньше числа процессов.
3. **Корректность результатов** — полное соответствие последовательной и параллельной версий.
4. **Модульность и тестируемость** — код структурирован и покрыт валидационными, функциональными и производительными тестами.

### 8.2. Выявленные проблемы и возможные улучшения:
1. **Улучшение масштабируемости** — основным ограничивающим фактором являются затраты на распределение данных и обмен граничными столбцами, что заметно при использовании небольшого ядра свёртки $3×3$.
2. **Оптимизация коммуникационных операций** — исследовать возможность снижения затрат на копирование данных и передачи их другим процессам.
3. **Рассмотрение более эффективных реализаций** — рассмотреть реализации, где минимизированы накладные расходы на передачу данных между процессами.

В рамках данной работы успешно решена задача линейной фильтрации изображений с использованием ядра Гаусса $3×3$, реализованы и проанализированы последовательная и параллельная MPI-версии алгоритма. Параллельная реализация демонстрирует практическую эффективность и может служить основой для дальнейших оптимизаций и расширений.

## 9. Источники
1. Расставим точки над структурами C/C++ / Антон Буков // habr.com[сайт] — режим доступа: https://habr.com/ru/articles/142662/ свободный (дата обращения: 10.12.2025).
2. Документация по курсу «Параллельное программирование» // Parallel Programming Course URL: https://learning-process.github.io/parallel_programming_course/ru/index.html (дата обращения: 25.10.2025).
3. std::memcpy // cppreference.com[сайт] — режим доступа: https://en.cppreference.com/w/cpp/string/byte/memcpy свободный (дата обращения: 20.12.2025).
4. "Коллективные и парные взаимодействия" / Сысоев А. В // Лекции по дисциплине «Параллельное программирование для кластерных систем».

## 10. Приложение

### 10.1. Приложение №1 — Общие определения
Файл: `./common/include/common.hpp`.
```cpp
struct ImageMetadata {
  size_t height{};
  size_t width{};
  size_t channels{};

  bool operator==(const ImageMetadata &other) const {
    return height == other.height && width == other.width && channels == other.channels;
  }
  bool operator!=(const ImageMetadata &other) const {
    return !(*this == other);
  }
};

using ImageData = std::vector<uint8_t>;
using Image = std::pair<ImageMetadata, ImageData>;

using InType = Image;
using OutType = Image;
using TestType = std::tuple<std::string, InType>;
using BaseTask = ppc::task::Task<InType, OutType>;

constexpr std::array<std::array<double, 3>, 3> kGaussianKernel = {
    {{1.0 / 16, 2.0 / 16, 1.0 / 16}, {2.0 / 16, 4.0 / 16, 2.0 / 16}, {1.0 / 16, 2.0 / 16, 1.0 / 16}}};

```

### 10.2. Приложение №2 — Последовательная версия решения задачи 
#### 10.2.1. Заголовочный файл:
Файл: `./seq/ops_seq.hpp`.
```cpp
class OtcheskovSGaussFilterVertSplitSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OtcheskovSGaussFilterVertSplitSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  uint8_t ProcessPixel(size_t row, size_t col, size_t ch);

  bool is_valid_{};
};
```

#### 10.2.1. Файл реализации:
Файл: `./seq/ops_seq.cpp`.
```cpp
OtcheskovSGaussFilterVertSplitSEQ::OtcheskovSGaussFilterVertSplitSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool OtcheskovSGaussFilterVertSplitSEQ::ValidationImpl() {
  const auto &[metadata, data] = GetInput();

  is_valid_ = !data.empty() && (metadata.height > 0 && metadata.width > 0 && metadata.channels > 0) &&
              (data.size() == metadata.height * metadata.width * metadata.channels);

  return is_valid_;
}

bool OtcheskovSGaussFilterVertSplitSEQ::PreProcessingImpl() {
  return true;
}

bool OtcheskovSGaussFilterVertSplitSEQ::RunImpl() {
  if (!is_valid_) {
    return false;
  }

  const auto &[in_meta, in_data] = GetInput();
  auto &[out_meta, out_data] = GetOutput();
  out_meta = in_meta;
  out_data.resize(in_data.size());
  const auto &[height, width, channels] = in_meta;

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      for (size_t ch = 0; ch < channels; ++ch) {
        size_t out_idx = (((row * width) + col) * channels) + ch;
        out_data[out_idx] = ProcessPixel(row, col, ch);
      }
    }
  }
  return true;
}

uint8_t OtcheskovSGaussFilterVertSplitSEQ::ProcessPixel(size_t row, size_t col, size_t ch) {
  const auto &[in_meta, in_data] = GetInput();
  const auto &[height, width, channels] = in_meta;

  auto mirror_coord = [&](const size_t &current, const int &off, const size_t &size) -> size_t {
    int64_t pos = static_cast<int64_t>(current) + off;
    if (pos < 0) {
      return static_cast<size_t>(-pos - 1);
    }
    if (std::cmp_greater_equal(static_cast<size_t>(pos), size)) {
      return (2 * size) - static_cast<size_t>(pos) - 1;
    }
    return static_cast<size_t>(pos);
  };

  double sum = 0.0;
  for (int dy = 0; dy < 3; ++dy) {
    size_t src_y = mirror_coord(row, dy - 1, height);
    for (int dx = 0; dx < 3; ++dx) {
      size_t src_x = mirror_coord(col, dx - 1, width);
      double weight = kGaussianKernel.at(dy).at(dx);
      size_t src_idx = (((src_y * width) + src_x) * channels) + ch;
      sum += weight * in_data[src_idx];
    }
  }
  return static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
}

bool OtcheskovSGaussFilterVertSplitSEQ::PostProcessingImpl() {
  return true;
}
```

### 10.3. Приложение №3 — Параллельная версия решения задачи

#### 10.3.1. Заголовочный файл
Файл: `./mpi/ops_mpi.hpp`.
```cpp
class OtcheskovSGaussFilterVertSplitMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit OtcheskovSGaussFilterVertSplitMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void DistributeData();
  [[nodiscard]] std::pair<std::vector<int>, std::vector<int>> GetCountsAndDisplacements(const size_t &height,
                                                                                        const size_t &width,
                                                                                        const size_t &channels) const;

  void ExchangeBoundaryColumns();

  void ApplyGaussianFilter();
  uint8_t ProcessPixel(const size_t &row, const size_t &local_col, const size_t &ch, const size_t &height,
                       const size_t &channels);
  void CollectResults();

  bool is_valid_{};
  size_t proc_rank_{};
  size_t proc_num_{};
  size_t active_procs_{};

  size_t local_width_{};
  size_t start_col_{};
  size_t local_data_count_{};

  std::vector<uint8_t> local_data_;
  std::vector<uint8_t> extended_data_;
  std::vector<uint8_t> local_output_;
};
```

#### 10.3.2. Файл реализации
Файл: `./mpi/ops_mpi.cpp`.
```cpp
OtcheskovSGaussFilterVertSplitMPI::OtcheskovSGaussFilterVertSplitMPI(const InType &in) {
  int proc_rank{};
  int proc_num{};
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
  SetTypeOfTask(GetStaticTypeOfTask());

  proc_rank_ = static_cast<size_t>(proc_rank);
  proc_num_ = static_cast<size_t>(proc_num);
  if (proc_rank_ == 0) {
    GetInput() = in;
  }
}

bool OtcheskovSGaussFilterVertSplitMPI::ValidationImpl() {
  if (proc_rank_ == 0) {
    const auto &[metadata, data] = GetInput();
    is_valid_ = !data.empty() && (metadata.height > 0 && metadata.width > 0 && metadata.channels > 0) &&
                (data.size() == metadata.height * metadata.width * metadata.channels);
  }
  MPI_Bcast(&is_valid_, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid_;
}

bool OtcheskovSGaussFilterVertSplitMPI::PreProcessingImpl() {
  return true;
}

bool OtcheskovSGaussFilterVertSplitMPI::RunImpl() {
  if (!is_valid_) {
    return false;
  }

  auto &[metadata, data] = GetInput();
  MPI_Bcast(&metadata, sizeof(ImageMetadata), MPI_BYTE, 0, MPI_COMM_WORLD);

  DistributeData();
  ExchangeBoundaryColumns();

  local_data_.clear();
  local_data_.shrink_to_fit();

  ApplyGaussianFilter();
  CollectResults();
  return true;
}

bool OtcheskovSGaussFilterVertSplitMPI::PostProcessingImpl() {
  return true;
}

void OtcheskovSGaussFilterVertSplitMPI::DistributeData() {
  const auto &[in_meta, in_data] = GetInput();
  const auto &[height, width, channels] = in_meta;

  active_procs_ = std::min(proc_num_, width);

  if (proc_rank_ < active_procs_) {
    const size_t base_cols = width / active_procs_;
    const size_t remain = width % active_procs_;
    local_width_ = base_cols + (proc_rank_ < remain ? 1 : 0);
    start_col_ = (base_cols * proc_rank_) + std::min(proc_rank_, remain);
  } else {
    local_width_ = 0;
    start_col_ = 0;
  }

  local_data_count_ = height * local_width_ * channels;
  local_data_.resize(local_data_count_);

  if (proc_rank_ == 0) {
    const size_t base_cols = width / active_procs_;
    const size_t remain = width % active_procs_;
    const auto &[counts, displs] = GetCountsAndDisplacements(height, width, channels);

    std::vector<uint8_t> send_buffer(counts[active_procs_ - 1] + displs[active_procs_ - 1]);
    for (size_t proc = 0; proc < active_procs_; ++proc) {
      const size_t cols = base_cols + (proc < remain ? 1 : 0);
      const size_t start_col = (base_cols * proc) + std::min(proc, remain);
      uint8_t *buf_ptr = send_buffer.data() + displs[proc];

      for (size_t row = 0; row < height; ++row) {
        const size_t row_size = width * channels;
        const size_t row_offset = row * row_size;
        const size_t col_offset = start_col * channels;
        const uint8_t *src_row = in_data.data() + row_offset + col_offset;
        std::memcpy(buf_ptr, src_row, cols * channels);
        buf_ptr += cols * channels;
      }
    }

    MPI_Scatterv(send_buffer.data(), counts.data(), displs.data(), MPI_UINT8_T, local_data_.data(),
                 static_cast<int>(local_data_count_), MPI_UINT8_T, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, local_data_.data(), static_cast<int>(local_data_count_),
                 MPI_UINT8_T, 0, MPI_COMM_WORLD);
  }
}

std::pair<std::vector<int>, std::vector<int>> OtcheskovSGaussFilterVertSplitMPI::GetCountsAndDisplacements(
    const size_t &height, const size_t &width, const size_t &channels) const {
  std::vector<int> counts(proc_num_, 0);
  std::vector<int> displs(proc_num_, 0);

  const size_t base_cols = width / active_procs_;
  const size_t remain = width % active_procs_;

  int total_data = 0;
  for (size_t proc = 0; proc < active_procs_; ++proc) {
    const size_t cols = base_cols + (proc < remain ? 1 : 0);
    counts[proc] = static_cast<int>(height * cols * channels);
    displs[proc] = total_data;
    total_data += counts[proc];
  }

  return {counts, displs};
}

void OtcheskovSGaussFilterVertSplitMPI::ExchangeBoundaryColumns() {
  if (local_width_ == 0) {
    extended_data_.clear();
    extended_data_.shrink_to_fit();
    return;
  }

  const auto &[in_meta, in_data] = GetInput();
  const size_t &height = in_meta.height;
  const size_t &channels = in_meta.channels;
  const size_t col_size = height * channels;

  int left_proc = MPI_PROC_NULL;
  int right_proc = MPI_PROC_NULL;

  if (proc_rank_ > 0 && proc_rank_ < active_procs_) {
    left_proc = static_cast<int>(proc_rank_) - 1;
  }
  if (proc_rank_ < active_procs_ - 1) {
    right_proc = static_cast<int>(proc_rank_) + 1;
  }

  std::vector<uint8_t> left_col(col_size);
  std::vector<uint8_t> right_col(col_size);
  std::vector<uint8_t> recv_left(col_size);
  std::vector<uint8_t> recv_right(col_size);

  for (size_t i = 0; i < height; ++i) {
    const size_t row_off = i * local_width_ * channels;
    const size_t dst_offset = i * channels;

    std::memcpy(&left_col[dst_offset], &local_data_[row_off], channels);
    std::memcpy(&right_col[dst_offset], &local_data_[row_off + ((local_width_ - 1) * channels)], channels);
  }

  if (left_proc != MPI_PROC_NULL) {
    MPI_Sendrecv(left_col.data(), static_cast<int>(col_size), MPI_UINT8_T, left_proc, 0, recv_right.data(),
                 static_cast<int>(col_size), MPI_UINT8_T, left_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  if (right_proc != MPI_PROC_NULL) {
    MPI_Sendrecv(right_col.data(), static_cast<int>(col_size), MPI_UINT8_T, right_proc, 1, recv_left.data(),
                 static_cast<int>(col_size), MPI_UINT8_T, right_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  const size_t ext_width = local_width_ + 2;
  extended_data_.resize(in_meta.height * ext_width * channels);

  for (size_t i = 0; i < in_meta.height; ++i) {
    uint8_t *ext_row = &extended_data_[i * ext_width * channels];
    const uint8_t *loc_row = &local_data_[i * local_width_ * channels];

    if (proc_rank_ == 0) {
      std::memcpy(ext_row, loc_row, channels);
    } else {
      std::memcpy(ext_row, &recv_right[i * channels], channels);
    }

    std::memcpy(ext_row + channels, loc_row, local_width_ * channels);

    if (proc_rank_ == active_procs_ - 1) {
      const uint8_t *last_col = &loc_row[(local_width_ - 1) * channels];
      std::memcpy(ext_row + ((ext_width - 1) * channels), last_col, channels);
    } else {
      std::memcpy(ext_row + ((ext_width - 1) * channels), &recv_left[i * channels], channels);
    }
  }
}

void OtcheskovSGaussFilterVertSplitMPI::ApplyGaussianFilter() {
  if (local_width_ == 0) {
    return;
  }
  const auto &[in_meta, in_data] = GetInput();
  local_output_.resize(local_data_count_);

  for (size_t row = 0; row < in_meta.height; ++row) {
    for (size_t local_col = 0; local_col < local_width_; ++local_col) {
      for (size_t ch = 0; ch < in_meta.channels; ++ch) {
        const size_t out_idx = ((row * local_width_ + local_col) * in_meta.channels) + ch;
        local_output_[out_idx] = ProcessPixel(row, local_col, ch, in_meta.height, in_meta.channels);
      }
    }
  }
}

uint8_t OtcheskovSGaussFilterVertSplitMPI::ProcessPixel(const size_t &row, const size_t &local_col, const size_t &ch,
                                                        const size_t &height, const size_t &channels) {
  auto mirror_coord = [&](const size_t &current, int off, const size_t &size) -> size_t {
    int64_t pos = static_cast<int64_t>(current) + off;
    if (pos < 0) {
      return static_cast<size_t>(-pos - 1);
    }
    if (std::cmp_greater_equal(static_cast<size_t>(pos), size)) {
      return (2 * size) - static_cast<size_t>(pos) - 1;
    }
    return static_cast<size_t>(pos);
  };

  double sum = 0.0;
  const size_t extended_width = local_width_ + 2;
  const size_t ext_col = local_col + 1;

  for (int ky = 0; ky < 3; ++ky) {
    const size_t data_row = mirror_coord(row, ky - 1, height);

    for (int kx = 0; kx < 3; ++kx) {
      const size_t data_col = ext_col + kx - 1;
      const size_t idx = ((data_row * extended_width + data_col) * channels) + ch;
      sum += extended_data_[idx] * kGaussianKernel.at(ky).at(kx);
    }
  }
  return static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
}

void OtcheskovSGaussFilterVertSplitMPI::CollectResults() {
  const auto &[in_meta, in_data] = GetInput();
  const auto &[height, width, channels] = in_meta;
  const size_t base_cols = width / active_procs_;
  const size_t remain = width % active_procs_;
  const size_t row_size = width * channels;

  std::vector<int> counts(proc_num_, 0);
  std::vector<int> displs(proc_num_, 0);

  int total_data = 0;
  for (size_t proc = 0; proc < active_procs_; ++proc) {
    const size_t cols = base_cols + (proc < remain ? 1 : 0);
    counts[proc] = static_cast<int>(height * cols * channels);
    displs[proc] = total_data;
    total_data += counts[proc];
  }

  if (proc_rank_ == 0) {
    auto &[out_meta, out_data] = GetOutput();
    out_meta = in_meta;
    out_data.resize(in_data.size());

    std::vector<uint8_t> recv_buffer(total_data);
    MPI_Gatherv(local_output_.data(), static_cast<int>(local_data_count_), MPI_UINT8_T, recv_buffer.data(),
                counts.data(), displs.data(), MPI_UINT8_T, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < height; ++i) {
      int buffer_offset = 0;
      for (size_t proc = 0; proc < active_procs_; ++proc) {
        const size_t cols = base_cols + (proc < remain ? 1 : 0);
        const size_t start_col = (base_cols * proc) + std::min(proc, remain);
        const uint8_t *src = recv_buffer.data() + static_cast<size_t>(buffer_offset) + (i * cols * channels);
        uint8_t *dst = out_data.data() + (i * row_size) + (start_col * channels);
        std::memcpy(dst, src, cols * channels);

        buffer_offset += counts[proc];
      }
    }
  } else {
    MPI_Gatherv(local_output_.data(), static_cast<int>(local_data_count_), MPI_UINT8_T, nullptr, nullptr, nullptr,
                MPI_UINT8_T, 0, MPI_COMM_WORLD);
  }
}
```

### 10.4. Приложение №4 — функциональные и валидационные тесты
Файл: `./tests/functional/main.cpp`.
```cpp
namespace {
InType ApplyGaussianFilter(const InType &input) {
  const auto &[metadata, image_data] = input;
  const auto &[height, width, channels] = metadata;
  const auto &data = image_data;

  ImageMetadata out_metadata = metadata;
  ImageData out_data = std::vector<uint8_t>(data.size());

  auto mirror_coord = [](size_t curr, int off, size_t size) {
    int64_t pos = static_cast<int64_t>(curr) + off;
    if (pos < 0) {
      return static_cast<size_t>(-pos - 1);
    }
    if (std::cmp_greater_equal(static_cast<size_t>(pos), size)) {
      return static_cast<size_t>((2 * size) - pos - 1);
    }
    return static_cast<size_t>(pos);
  };

  const size_t row_stride = width * channels;

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      for (size_t ch = 0; ch < channels; ++ch) {
        double sum = 0.0;
        for (int dy = -1; dy <= 1; ++dy) {
          size_t src_y = mirror_coord(row, dy, height);
          for (int dx = -1; dx <= 1; ++dx) {
            size_t src_x = mirror_coord(col, dx, width);
            size_t src_idx = (src_y * row_stride) + (src_x * channels) + ch;
            sum += data[src_idx] * kGaussianKernel.at(dy + 1).at(dx + 1);
          }
        }
        size_t out_idx = (row * row_stride) + (col * channels) + ch;
        out_data[out_idx] = static_cast<uint8_t>(std::clamp(std::round(sum), 0.0, 255.0));
      }
    }
  }
  return {out_metadata, out_data};
}

InType CreateGradientImage(ImageMetadata img_metadata) {
  ImageData img_data;
  const auto &[width, height, channels] = img_metadata;
  img_data.resize(width * height * channels);

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      for (size_t ch = 0; ch < channels; ++ch) {
        const size_t idx = (row * width * channels) + (col * channels) + ch;
        img_data[idx] = static_cast<uint8_t>((col * 2 + row + ch * 50) % 256);
      }
    }
  }

  return {img_metadata, img_data};
}

InType LoadRgbImage(const std::string &img_path) {
  int width = -1;
  int height = -1;
  int channels_in_file = -1;

  unsigned char *data = stbi_load(img_path.c_str(), &width, &height, &channels_in_file, STBI_rgb);
  if (data == nullptr) {
    throw std::runtime_error("Failed to load image '" + img_path + "': " + std::string(stbi_failure_reason()));
  }

  ImageMetadata img_metadata;
  ImageData img_data;
  img_metadata.width = static_cast<size_t>(width);
  img_metadata.height = static_cast<size_t>(height);
  img_metadata.channels = STBI_rgb;
  const auto bytes = img_metadata.width * img_metadata.height * img_metadata.channels;
  img_data.assign(data, data + bytes);
  stbi_image_free(data);
  return {img_metadata, img_data};
}

}  // namespace

class OtcheskovSGaussFilterVertSplitValidationTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.second.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

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
    task_->GetInput() = std::get<1>(params);
    ExecuteTaskPipeline();
  }

  void ExecuteTaskPipeline() {
    EXPECT_FALSE(task_->Validation());
    task_->PreProcessing();
    task_->Run();
    task_->PostProcessing();
  }

 private:
  InType input_data_;
  ppc::task::TaskPtr<InType, OutType> task_;
};

class OtcheskovSGaussFilterVertSplitFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const InType &input_test_data = std::get<1>(params);
    input_img_ = CreateGradientImage(input_test_data.first);
    expect_img_ = ApplyGaussianFilter(input_img_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (!ppc::util::IsUnderMpirun()) {
      return expect_img_ == output_data;
    }

    int proc_rank{};
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    if (proc_rank == 0) {
      return expect_img_ == output_data;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_img_;
  }

 private:
  InType input_img_;
  InType expect_img_;
};

class OtcheskovSGaussFilterVertSplitRealTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string filename = std::get<0>(test_param);

    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
      filename = filename.substr(0, dot_pos);
    }

    return filename;
  }

 protected:
  void SetUp() override {
    try {
      const TestType &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
      const std::string &filename = std::get<0>(params);
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_otcheskov_s_gauss_filter_vert_split, filename);
      input_img_ = LoadRgbImage(abs_path);
      expect_img_ = ApplyGaussianFilter(LoadRgbImage(abs_path));
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (!ppc::util::IsUnderMpirun()) {
      return expect_img_ == output_data;
    }

    int proc_rank{};
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    if (proc_rank == 0) {
      return expect_img_ == output_data;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_img_;
  }

 private:
  InType input_img_;
  InType expect_img_;
};

namespace {

const std::array<TestType, 5> kTestValidParam = {
    {{"empty_data", {ImageMetadata{.height = 3, .width = 3, .channels = 3}, ImageData{}}},
     {"image_3x3x1_wrong_size",
      {ImageMetadata{.height = 4, .width = 4, .channels = 3}, ImageData{10, 12, 14, 15, 16, 17, 18, 19, 50}}},
     {"image_3x3x1_wrong_height",
      {ImageMetadata{.height = 0, .width = 3, .channels = 1}, ImageData{10, 12, 14, 15, 16, 17, 18, 19, 50}}},
     {"image_3x3x1_wrong_width",
      {ImageMetadata{.height = 3, .width = 0, .channels = 1}, ImageData{10, 12, 14, 15, 16, 17, 18, 19, 50}}},
     {"image_3x3x1_wrong_channel",
      {ImageMetadata{.height = 3, .width = 3, .channels = 0}, ImageData{10, 12, 14, 15, 16, 17, 18, 19, 50}}}}};

const std::array<TestType, 8> kTestFuncParam = {
    {{"image_15x1x1", {ImageMetadata{.height = 15, .width = 1, .channels = 1}, ImageData{}}},
     {"image_3x3x1", {ImageMetadata{.height = 3, .width = 3, .channels = 1}, ImageData{}}},
     {"image_3x3x3", {ImageMetadata{.height = 3, .width = 3, .channels = 3}, ImageData{}}},
     {"image_4x4x1", {ImageMetadata{.height = 4, .width = 4, .channels = 1}, ImageData{}}},
     {"image_10x20x3", {ImageMetadata{.height = 10, .width = 20, .channels = 3}, ImageData{}}},
     {"border_test_9x9", {ImageMetadata{.height = 9, .width = 9, .channels = 1}, ImageData{}}},
     {"border_test_10x10", {ImageMetadata{.height = 10, .width = 10, .channels = 1}, ImageData{}}},
     {"sharp_vertical_lines_15x15", {ImageMetadata{.height = 15, .width = 15, .channels = 3}, ImageData{}}}}};

const std::array<TestType, 2> kTestRealParam = {
    {{"chess.jpg", {ImageMetadata{}, ImageData{}}}, {"gradient.jpg", {ImageMetadata{}, ImageData{}}}}};

const auto kTestValidTasksList = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitMPI, InType>(
                                                    kTestValidParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split),
                                                ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitSEQ, InType>(
                                                    kTestValidParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split));

const auto kTestFuncTasksList = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitMPI, InType>(
                                                   kTestFuncParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split),
                                               ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitSEQ, InType>(
                                                   kTestFuncParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split));

const auto kTestRealTasksList = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitMPI, InType>(
                                                   kTestRealParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split),
                                               ppc::util::AddFuncTask<OtcheskovSGaussFilterVertSplitSEQ, InType>(
                                                   kTestRealParam, PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split));

const auto kGtestValidValues = ppc::util::ExpandToValues(kTestValidTasksList);
const auto kGtestFuncValues = ppc::util::ExpandToValues(kTestFuncTasksList);
const auto kGtestRealValues = ppc::util::ExpandToValues(kTestRealTasksList);

const auto kValidFuncTestName = OtcheskovSGaussFilterVertSplitValidationTestsProcesses::PrintFuncTestName<
    OtcheskovSGaussFilterVertSplitValidationTestsProcesses>;

const auto kFuncTestName = OtcheskovSGaussFilterVertSplitFuncTestsProcesses::PrintFuncTestName<
    OtcheskovSGaussFilterVertSplitFuncTestsProcesses>;

const auto kRealTestName = OtcheskovSGaussFilterVertSplitRealTestsProcesses::PrintFuncTestName<
    OtcheskovSGaussFilterVertSplitRealTestsProcesses>;

TEST_P(OtcheskovSGaussFilterVertSplitValidationTestsProcesses, GaussFilterVertSplitValidation) {
  ExecuteTest(GetParam());
}

TEST_P(OtcheskovSGaussFilterVertSplitFuncTestsProcesses, GaussFilterVertSplitFunc) {
  ExecuteTest(GetParam());
}

TEST_P(OtcheskovSGaussFilterVertSplitRealTestsProcesses, GaussFilterVertSplitReal) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(GaussFilterVertSplitValidation, OtcheskovSGaussFilterVertSplitValidationTestsProcesses,
                         kGtestValidValues, kValidFuncTestName);

INSTANTIATE_TEST_SUITE_P(GaussFilterVertSplitFunc, OtcheskovSGaussFilterVertSplitFuncTestsProcesses, kGtestFuncValues,
                         kFuncTestName);

INSTANTIATE_TEST_SUITE_P(GaussFilterVertSplitReal, OtcheskovSGaussFilterVertSplitRealTestsProcesses, kGtestRealValues,
                         kRealTestName);

}  // namespace
```

### 10.5. Приложение №5 — производительные тесты
Файл: `./tests/performance/main.cpp`.
```cpp
namespace {
InType CreateGradientImage(ImageMetadata img_metadata) {
  ImageData img_data;
  const auto &[height, width, channels] = img_metadata;
  const size_t pixel_count = width * height * channels;
  img_data.resize(pixel_count);

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      for (size_t ch = 0; ch < channels; ++ch) {
        const size_t idx = (row * width * channels) + (col * channels) + ch;
        img_data[idx] = static_cast<uint8_t>((col * 2 + row + ch * 50) % 256);
      }
    }
  }

  return {img_metadata, img_data};
}

}  // namespace

class OtcheskovSGaussFilterVertSplitPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr size_t kMatrixSize = 10000;
  InType input_img_;

  void SetUp() override {
    input_img_ = CreateGradientImage(ImageMetadata{.height = kMatrixSize, .width = kMatrixSize, .channels = 3});
  }

  bool CheckTestOutputData(OutType &output_img) final {
    bool is_checked = false;
    if (!ppc::util::IsUnderMpirun()) {
      is_checked = output_img.first == input_img_.first;
    } else {
      int proc_rank{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
      if (proc_rank == 0) {
        is_checked = output_img.first == input_img_.first;
      } else {
        is_checked = true;
      }
    }
    return is_checked;
  }

  InType GetTestInputData() final {
    return input_img_;
  }
};

TEST_P(OtcheskovSGaussFilterVertSplitPerfTests, RunPerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OtcheskovSGaussFilterVertSplitMPI, OtcheskovSGaussFilterVertSplitSEQ>(
        PPC_SETTINGS_otcheskov_s_gauss_filter_vert_split);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OtcheskovSGaussFilterVertSplitPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfTests, OtcheskovSGaussFilterVertSplitPerfTests, kGtestValues, kPerfTestName);
```