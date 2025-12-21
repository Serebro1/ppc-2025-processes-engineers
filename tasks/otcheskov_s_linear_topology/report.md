# Линейная топология процессов (Линейка)

- Студент: Отческов Семён Андреевич, группа 3823Б1ПР1
- Технологии: SEQ | MPI
- Вариант: 6


## 1. Введение
- В параллельном программировании важны связи между процессами для обмена и распределения данных. Для организации структуры узлов и линий в сети используются топологии. Их можно представить в виде графа, где вершины — процессоры (процессы), а дуги — каналы связи.
- С помощью MPI можно организовать логические представления любой виртуальной топологии.
- В данной работе будет рассмотрена простая линейная топология (линейка), где каждый процессор (процесс), кроме первого и последнего, имеет линии связи только с двумя соседними (с предыдущими и последующими) процессорами.
- Будут представлены две реализации: последовательная (SEQ) и параллельная (MPI).
  - Так как реализация топологии подразумевает использование MPI, то последовательная версия будет представлена, как граф с одним звеном и циклической связью.
  - Участвовать в сравнении производительности с MPI версией она не будет.

- **Цель работы:** реализация виртуальной линейной топологии средствами MPI.


## 2. Постановка задачи
**Формальная постановка:**
- Реализовать виртуальную линейную топологию, используя возможности MPI по работе с коммуникаторами и топологиями и обеспечить возможность передачи данных от любого выбранного процесса любому другому процессу.

**Входные данные:**
- Заголовок сообщения (MessageHeader).
- Данные (MessageData).

Далее обе структуры вместе будут называться "сообщением".

**Выходные данные:**
- Процесс-получатель: сообщение с вектором данных, полученный от процесса-отправителя.
- Остальные-процессы: сообщение с поднятым флагом подтверждения получения, который прислал процесс-получатель.

**Ограничения:**
- Источник и получатель должны быть в допустимом диапазоне рангов.
- Вектор должен содержать хотя бы один элемент.
- Запрещается использование MPI_Cart_Create и MPI_Graph_Create для создания коммуникатора с нужной топологией.


## 3. Описание алгоритма (последовательного)

### 3.1. Этапы выполнения задачи
**1. Валидация данных (`ValidationImpl`):**
- Проверка корректности полей (метаданных) заголовка.
- Проверка на пустоту вектора.
- Проверка на заявленный в заголовке размер входного вектора с фактическим.

**2. Предобработка данных (`PreProcessingImpl`):**
- Задача не требует предобработки, поэтому данный этап пропускается.

**3. Вычисления (`RunImpl`):**
- Так как процесс один, то просто копируются входные данные в выходные.
- Ставится флаг "доставки" в true.

**4. Постобработка данных (`PostProcessingImpl`):**
- Аналогична предобработке.

### 3.2. Сложность алгоритма:
- Временная сложность: `O(1)` — прямая передача данных.
- Пространственная сложность: `O(N)` — хранение вектора данных произвольной длины N.


## 4. Схема распараллеливания алгоритма с помощью MPI

### 4.1. Структура сообщения
- Сообщение состоит из заголовка (MessageHeader), содержащий:
  - Флаг доставки (delivered).
  - Номер процесса-источника.
  - Номер процесса-получателя.
  - Размер данных.
- Данные (MessageData).
- Заголовок передаётся отдельно от данных последовательно (MPI_BYTE).
- Данные передаются как вектор целых чисел.

### 4.2. Топология коммуникаций
- Линейная топология (процессы связаны через `MPI_COMM_WORLD`), организуемая динамически на основе рангов источника и получателя.
- Направление передачи определяется сравнением рангов источника и получателя.
- Участвуют только процессы, находящиеся на пути от источника к получателю

### 4.3. Паттерны коммуникации
- В MPI реализации используются коллективные блокирующие функции.
- Функция `MPI_Bcast`:
  - Применяется в валидации, чтобы каждый процесс прошёл проверку аналогично процессу-источнику.
- Функции `MPI_Send` и `MPI_Recv` используются для:
  - Двусторонняя передача с подтверждением:
    1. Передача сообщения от источника к получателю.
    2. Передача подтверждения от получателя к источнику.
  - Прямая связь для соседних процессов:
    - Каждый процесс передает сообщение следующему указанному процессу в цепочке.

### 4.4. Процесс передачи:
1. **Инициализация MPI** — получение ранга и числа процессов.
2. **Валидация** — проверка корректности входных данных на процессе-источнике и рассылка результата проверки всем процессам.
3. **Подготовка сообщения** — Процесс-источник подготавливает заголовок и данные для отправки.
4. **Составление цепочки** — определяем участников цепочки, направление движения и соседей процессов.
5. **Передача сообщения**— последовательная передача по цепочке процессов.
5. **Подтверждение доставки** — обратная передача заголовка сообщения с поднятым флагом доставки.
6. **Синхронизация процессов** — для корректной работы все выделенные процессы синхронизируются в конце всей передачи.

### 4.5. Особые случаи:
- **Локальная передача** (`source == destination`):
  - Флаг доставки устанавливается в 1 и сообщение никому не передаётся.
- **Непосредственная передача** (процессы-соседи):
  - Минимальное количество промежуточных процессов.
- **Длинная цепочка**:
  - Маскимальное количество промежуточных процессов.


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
- `./tests/performance/main.cpp` — реализация производительных тестов ([см. Приложение №5](#105-приложение-5--производительные-тесты)).

#### 5.1.2. Ключевые классы
- `OtcheskovSLinearTopologySEQ` — последовательная версия.
- `OtcheskovSLinearTopologyMPI` — параллельная версия.
- `OtcheskovSLinearTopologyFuncTests` — функциональные тесты.
- `OtcheskovSLinearTopologyFuncTestsValidation` — валидационные тесты.
- `OtcheskovSLinearTopologyPerfTests` — производительные тесты.

#### 5.1.3. Основные методы
- `ValidationImpl` — валидация входных данных.
- `PreProcessingImpl` — препроцессинг, не используется.
- `RunImpl` — основная логика передачи сообщения.
- `PostProcessingImpl` — постпроцессинг, не используется.

#### 5.1.4. Основные структуры
- `std::pair<MessageHeader, MessageData>` — сообщение.
- `MessageHeader` — заголовок сообщения.
- `MessageData` — данные сообщения.

### 5.2. Описание реализации последовательной версии (заглушка)
Полная реализация данной версии указана в [Приложении №2](#102-приложение-2--последовательная-версия-решения-задачи)

#### 5.2.1. ValidationImpl
- Заголовок сообщения:
  - Номер процесса-источника должен находиться в допустимом диапазоне [минимальный номер процесса коммутатора, максимальный номер процесса коммутатора].
  - Номер процесса-отправителя должен находиться в допустимом диапазоне [минимальный номер процесса коммутатора, максимальный номер процесса коммутатора].
  - Флаг доставки должен быть установлен в `false`.
- Данные сообщения:
  - Данные не должны быть пустыми.
  - Фактический размер данных должен соответствовать размеру, указанному в заголовке сообщения.

#### 5.2.2. RunImpl
- Копирование входного сообщения в выходное.
- Установка флага доставки выходного сообщения.

### 5.3. Реализация параллельной версии

#### 5.3.1. ValidationImpl
- Проверка выполняется на процессе-источнике аналогично последовательной версии.
- Результат рассылается всем процессам через `MPI_Bcast`.
- Единая проверка для всех процессов.

#### 5.3.2. RunImpl
1. Определение участников цепочки и направления движения.
  - Процессы, не участвующие в передаче, имеют пустые сообщения. 
2. Прямая передача сообщения.
3. Передача подтверждения.
4. Проверка результатов отправки и получения:
  - Процесс-источник: проверка флага доставки.
  - Процесс-получатель: проверка получения данных.
5. Синхронизация процессов.
  - Для корректных измерения все выделенные программе процессы синхронизируются с помощью `MPI_Barrier`.

#### 5.3.3. Вспомогательные методы
- `ForwardMessageToDest` — прямая передача по цепочке процессов.
  - Для идентификации операции используется тег kMessageTag.
- `HandleConfirmToSource` — обработка подтверждения от процесса-получателя.
  - Для идентификации операции используется тег kConfirmTag.
- `SendMessageLinear` — подготовка процессов к передаче и запуск двух выше перечисленных функций для отправления данных и получения ответа.

### 5.4. Использование памяти
- Последовательная версия: `O(N)` — входной вектор произвольной длины `N`.
- Параллельная версия: 
  - Процесс-источник: `O(N)` — хранит исходный массив.
  - Промежуточные процессы: `O(N)` — временное хранение данных при передаче.
  - Процесс-получатель: `O(N)` — полученные данные.

### 5.5. Допущения и крайние случаи
- Все процессы запускаются в рамках одного MPI-коммуникатора.
- Процесс-источник является валидирующим процессом.
- Данные передаются только от источника к получателю.
- Подтверждение доставки обязательно.

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
- **Данные:** тестовые сообщения генерируются программно.


## 7. Результаты и обсуждение

### 7.1. Корректность

Корректность задачи для обоих версий проверена с помощью набора параметризированных тестов Google Test.
Реализации функциональных и валидационных тестов находятся в [Приложении №10.4](#104-приложение-4--функциональные-и-валидационные-тесты)

#### 7.1.1. Функциональные тесты
- Локальная передача (src == dest):
  - Тест с 5 элементами
  - Тест с 1 элементом
- Переедача между двумя процессами:
  - Слева-направо (0 --> 1).
  - Справа-налево (1 --> 0).
  - Самим себе (0 --> 0 и 1 --> 1).
- Передача между четырьмя процессами:
  - Между несоседними (краевыми) процессами (0 --> 3 и 3 --> 0).
  - Соседние процессы (1 --> 2 и 2 --> 1).

#### 7.1.2. Валидационные тесты
- Некорректный источник или получатель:
  - Отрицательные ранги.
  - Ранги за пределами числа процессов.
- Некорректный размер данных:
  - Несоответствие заголовка и фактических данных.
  - Пустые данные при ненулевом размере.
- Некорректный флаг доставки:
  - Флаг доставки заранее предустановлен.

#### 7.1.3. Механизм проверки
- Для SEQ и MPI реализаций написаны разные тесты.
- Для SEQ-версии проверяется только флаг доставки.
- Для MPI-версии проверяется:
  - Флаг доставки данных на процессе-источнике.
  - Корректность доставки данных на процессе-получателе.
- Для некорректных сценариев проверяется провал валидации (`ValidationImpl()`).


### 7.2. Производительные тесты

#### 7.2.1. Методология тестирования
- **Данные:** вектор из 128 миллионов целых чисел
- **Режимы:**
  - **pipeline** — запуск и измерение времени всех этапов алгоритма (`Validation -> PreProcessing -> Run -> PostProcessing`).
  - **task_run** — запуск всех этапов алгоритма, но измеряется время только на этапе `Run`.
- **Производительность** мерилась только в режиме `task_run`.
- **Метрики:** число процессов, абсолютное время выполнения task_run, ускорение, эффективность.
- **Сравнение:**
  - Сравнивается данные с результатами при минимальном числе процессов (2).

#### 7.2.2. Результаты тестирования на 128 миллионов элементов

**Windows:**
| Режим | Процесс-источник | Процесс-получатель | Число процессов    | Время, s | Ускорение | Эффективность |
| ----- | ---------------- | ------------------ | ------------------ | -------- | --------- | ------------- |
| mpi   | 0                | 1                  | 2                  | 0.757685 | 1.0000    | 50.0%         |
| mpi   | 2                | 0                  | 3                  | 0.893686 | 0.8478    | 28.3%         |
| mpi   | 3                | 1                  | 3 (запуск на 4-ёх) | 0.894192 | 0.3684    | 28.3%         |
| mpi   | 0                | 3                  | 4                  | 1.049943 | 0.7216    | 18.0%         |
| mpi   | 5                | 2                  | 4 (запуск на 6-ти) | 1.034309 | 0.3684    | 18.3%         |
| mpi   | 0                | 5                  | 6                  | 1.342075 | 0.3684    |  9.4%         |


**WSL:**
| Режим | Процесс-источник | Процесс-получатель | Число процессов    | Время, s | Ускорение | Эффективность |
| ----- | ---------------- | ------------------ | ------------------ | -------- | --------- | ------------- |
| mpi   | 0                | 1                  | 2                  | 1.258171 | 1.0000    | 50.0%         |
| mpi   | 2                | 0                  | 3                  | 1.443387 | 0.8718    | 29.1%         |
| mpi   | 3                | 1                  | 3 (запуск на 4-ёх) | 1.480873 | 0.8496    | 28.3%         |
| mpi   | 0                | 3                  | 4                  | 1.702868 | 0.7391    | 18.5%         |
| mpi   | 5                | 2                  | 4 (запуск на 6-ти) | 1.665639 | 0.7555    | 18.9%         |
| mpi   | 0                | 5                  | 6                  | 2.147666 | 0.5862    |  9.8%         |

**\*GitHub:**
| Режим | Процесс-источник | Процесс-получатель | Число процессов | Время, s | Ускорение | Эффективность |
| ----- | ---------------- | ------------------ | --------------- | -------- | --------- | ------------- |
| mpi   | 0                | 1                  | 2               | 0.664711 | 1.0000    | 50.0%         |

*\*Результаты собирались на локальном форке из Github Actions*

### 7.3. Анализ результатов
- MPI версия:
  - Корректно реализует линейную топологию.
  - Обеспечивает подтверждение доставки.

- Ограничение масштабируемости:
  - Время передачи линейно зависит от расстояния между процессами.
  - Промежуточные процесы становятся узкими местами.
  - Большие объемы данных требут значительных буферов.


## 8. Заключения

### 8.1. Достигнутые результаты:
1. **Реализация линейной топологии** — корректная передача сообщений по цепочке процессов
2. **Передача с подтверждением** — гарантированная доставка с подтверждением
3. **Обработка крайних случаев** — локальная передача, некорректные параметры
4. **Модульность и тестируемость** — код структурирован и покрыт функциональными и валидационными тестами.

В рамках данной работы успешно решена задача реализации линейной топологии процессов, разработаны последовательная и параллельная версии алгоритма, проведено тестирование корректности и производительности.

## 9. Источники
1. Modern problems of informatic // www.nsc.ru[сайт] — режим доступа: http://www.nsc.ru/win/elbib/data/show_page.dhtml?77+783 свободный (дата обращения: 1.12.2025)
2. Расставим точки над структурами C/C++ / Антон Буков // habr.com[сайт] — режим доступа: https://habr.com/ru/articles/142662/ свободный (дата обращения: 10.12.2025) — Загл. с экрана.
3. Документация по курсу «Параллельное программирование» // Parallel Programming Course URL: https://learning-process.github.io/parallel_programming_course/ru/index.html (дата обращения: 1.12.2025).
4. "Основы MPI" / Сысоев А. В // Лекции по дисциплине «Параллельное программирование для кластерных систем».
5. "Виртуальные топологии" / Сысоев А. В // Лекции по дисциплине «Параллельное программирование для кластерных систем».
## 10. Приложение

### 10.1. Приложение №1 — Общие определения
Файл: `./common/include/common.hpp`.
```cpp
struct MessageHeader {
  int delivered{};
  int src{};
  int dest{};
  int data_size{};
};

using MessageData = std::vector<int>;
using Message = std::pair<MessageHeader, MessageData>;

using InType = Message;
using OutType = Message;
using TestType = std::tuple<Message, int>;
using BaseTask = ppc::task::Task<InType, OutType>;
```

### 10.2. Приложение №2 — Последовательная версия решения задачи 
#### 10.2.1. Заголовочный файл:
Файл: `./seq/ops_seq.hpp`.
```cpp
class OtcheskovSLinearTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OtcheskovSLinearTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
```

#### 10.2.1. Файл реализации:
Файл: `./seq/ops_seq.cpp`.
```cpp
OtcheskovSLinearTopologySEQ::OtcheskovSLinearTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool OtcheskovSLinearTopologySEQ::ValidationImpl() {
  const auto &header = GetInput().first;
  const auto &data = GetInput().second;
  return header.src >= 0 && header.dest >= 0 && header.delivered == 0 && !data.empty() &&
         static_cast<size_t>(header.data_size) == data.size();
}

bool OtcheskovSLinearTopologySEQ::PreProcessingImpl() {
  return true;
}

bool OtcheskovSLinearTopologySEQ::RunImpl() {
  GetOutput() = GetInput();
  GetOutput().first.delivered = 1;
  return true;
}

bool OtcheskovSLinearTopologySEQ::PostProcessingImpl() {
  return true;
}
```

### 10.3. Приложение №3 — Параллельная версия решения задачи

#### 10.3.1. Заголовочный файл
Файл: `./mpi/ops_mpi.hpp`.
```cpp
class OtcheskovSLinearTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit OtcheskovSLinearTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] static Message ForwardMessageToDest(const Message &initial_msg, int prev, int next, bool is_src,
                                                    bool is_dest);
  [[nodiscard]] static Message HandleConfirmToSource(Message &current_msg, int prev, int next, bool is_src,
                                                     bool is_dest);
  [[nodiscard]] Message SendMessageLinear(const Message &msg) const;

  int proc_rank_{};
  int proc_num_{};
};
```

#### 10.3.2. Файл реализации
Файл: `./mpi/ops_mpi.cpp`.
```cpp
namespace {
constexpr int kMessageTag = 100;
constexpr int kConfirmTag = 200;
constexpr int kDataTagOffset = 1;
}  // namespace

OtcheskovSLinearTopologyMPI::OtcheskovSLinearTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);

  GetInput() = in;
  if (proc_rank_ != in.first.src) {
    GetInput().second.clear();
    GetInput().second.shrink_to_fit();
  }
}

bool OtcheskovSLinearTopologyMPI::ValidationImpl() {
  const auto &[header, data] = GetInput();
  if (header.src < 0 || header.src >= proc_num_) {
    return false;
  }

  bool is_valid = false;

  if (proc_rank_ == header.src) {
    is_valid = (header.dest >= 0 && header.dest < proc_num_ && header.delivered == 0 && !data.empty() &&
                static_cast<size_t>(header.data_size) == data.size());
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, GetInput().first.src, MPI_COMM_WORLD);
  return is_valid;
}

bool OtcheskovSLinearTopologyMPI::PreProcessingImpl() {
  return true;
}

Message OtcheskovSLinearTopologyMPI::ForwardMessageToDest(const Message &initial_msg, int prev, int next, bool is_src,
                                                          bool is_dest) {
  Message current_msg;
  auto &[header, data] = current_msg;
  if (!is_src) {
    MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE, prev, kMessageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    data.resize(header.data_size);
    MPI_Recv(data.data(), header.data_size, MPI_INT, prev, kMessageTag + kDataTagOffset, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    current_msg = initial_msg;
  }

  if (!is_dest) {
    MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, next, kMessageTag, MPI_COMM_WORLD);
    MPI_Send(data.data(), header.data_size, MPI_INT, next, kMessageTag + kDataTagOffset, MPI_COMM_WORLD);
    if (!is_src) {
      data.clear();
      data.shrink_to_fit();
    }
  } else {
    header.delivered = 1;
  }
  return current_msg;
}

Message OtcheskovSLinearTopologyMPI::HandleConfirmToSource(Message &current_msg, int prev, int next, bool is_src,
                                                           bool is_dest) {
  auto& confirm_header = current_msg.first;
  if (is_dest) {
    MPI_Send(&confirm_header, sizeof(MessageHeader), MPI_BYTE, prev, kConfirmTag, MPI_COMM_WORLD);
  } else {
    MPI_Recv(&confirm_header, sizeof(MessageHeader), MPI_BYTE, next, kConfirmTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!is_src) {
      MPI_Send(&confirm_header, sizeof(MessageHeader), MPI_BYTE, prev, kConfirmTag, MPI_COMM_WORLD);
    }
  }
  return current_msg;
}

Message OtcheskovSLinearTopologyMPI::SendMessageLinear(const Message &msg) const {
  auto [header, data] = msg;
  header.delivered = 0;

  if (header.src == header.dest) {
    header.delivered = 1;
    return {header, std::move(data)};
  }

  const int direction = (header.dest > header.src) ? 1 : -1;
  const bool should_participate = (direction > 0 && proc_rank_ >= header.src && proc_rank_ <= header.dest) ||
                                  (direction < 0 && proc_rank_ <= header.src && proc_rank_ >= header.dest);

  if (!should_participate) {
    return {MessageHeader(), MessageData()};
  }

  const bool is_src = (proc_rank_ == header.src);
  const bool is_dest = (proc_rank_ == header.dest);

  const int prev = is_src ? MPI_PROC_NULL : proc_rank_ - direction;
  const int next = is_dest ? MPI_PROC_NULL : proc_rank_ + direction;

  Message current_msg = ForwardMessageToDest({header, std::move(data)}, prev, next, is_src, is_dest);
  // пересылка подтверждения
  return HandleConfirmToSource(current_msg, prev, next, is_src, is_dest);
}

bool OtcheskovSLinearTopologyMPI::RunImpl() {
  const auto &in_header = GetInput().first;
  const int src = in_header.src;
  const int dest = in_header.dest;

  if (src < 0 || src >= proc_num_) {
    return false;
  }

  MessageHeader msg_header;
  msg_header.src = src;
  msg_header.dest = dest;
  msg_header.delivered = 0;
  msg_header.data_size = 0;

  MessageData data;
  if (proc_rank_ == src) {
    data = GetInput().second;
    msg_header.data_size = static_cast<int>(data.size());
  }

  Message result_msg = SendMessageLinear({msg_header, data});

  bool check_passed = false;
  if (proc_rank_ == src) {
    check_passed = result_msg.first.delivered != 0;
  } else if (proc_rank_ == dest) {
    check_passed = !result_msg.second.empty();
  } else {
    check_passed = true;
  }
  GetOutput() = result_msg;
  MPI_Barrier(MPI_COMM_WORLD);
  return check_passed;
}

bool OtcheskovSLinearTopologyMPI::PostProcessingImpl() {
  return true;
}
```

### 10.4. Приложение №4 — функциональные и валидационные тесты
Файл: `./tests/functional/main.cpp`.
```cpp
class OtcheskovSLinearTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param).first.src) + "_" +
           std::to_string(std::get<0>(test_param).first.dest) + "_process_" + "test_" +
           std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_msg = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_msg) final {
    const auto &[in_header, in_data] = input_msg;
    const auto &[out_header, out_data] = output_msg;

    bool is_valid = false;
    if (!ppc::util::IsUnderMpirun()) {
      is_valid = out_header.delivered != 0;
    } else {
      int proc_rank{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
      if (proc_rank == in_header.src || proc_rank == in_header.dest) {
        is_valid = (in_data == out_data) && out_header.delivered != 0;
      } else {
        is_valid = true;
      }
    }
    return is_valid;
  }

  InType GetTestInputData() final {
    return input_msg;
  }

  InType input_msg;
};

class OtcheskovSLinearTopologyMpi2ProcTests : public OtcheskovSLinearTopologyFuncTests {
 protected:
  void SetUp() override {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "kMPI tests are not under mpirun\n";
      GTEST_SKIP();
    }
    OtcheskovSLinearTopologyFuncTests::SetUp();
  }
};

class OtcheskovSLinearTopologyMpi4ProcTests : public OtcheskovSLinearTopologyFuncTests {
 protected:
  void SetUp() override {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "kMPI tests are not under mpirun\n";
      GTEST_SKIP();
    }
    OtcheskovSLinearTopologyFuncTests::SetUp();
  }
};

class OtcheskovSLinearTopologyFuncTestsValidation : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return FormatNumber(std::get<0>(test_param).first.src) + "_" + FormatNumber(std::get<0>(test_param).first.dest) +
           "_process_" + "test_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  bool CheckTestOutputData(OutType &output_msg) final {
    return output_msg.first.delivered == 0;
  }

  InType GetTestInputData() final {
    return input_msg_;
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
    ExecuteTaskPipeline();
  }

  void ExecuteTaskPipeline() {
    EXPECT_FALSE(task_->Validation());
    task_->PreProcessing();
    task_->Run();
    task_->PostProcessing();
  }

 private:
  static std::string FormatNumber(int value) {
    if (value >= 0) {
      return std::to_string(value);
    }
    return "minus_" + std::to_string(-value);
  }

  InType input_msg_;
  ppc::task::TaskPtr<InType, OutType> task_;
};

namespace {

TEST_P(OtcheskovSLinearTopologyMpi2ProcTests, Mpi2ProcsTests) {
  int proc_nums{};
  MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);
  if (proc_nums < 2) {
    std::cerr << "Tests should run on 2 or more processes\n";
  } else {
    ExecuteTest(GetParam());
  }
}

TEST_P(OtcheskovSLinearTopologyMpi4ProcTests, Mpi4ProcsTests) {
  int proc_nums{};
  MPI_Comm_size(MPI_COMM_WORLD, &proc_nums);
  if (proc_nums < 4) {
    std::cerr << "Tests should run on 4 or more processes\n";
  } else {
    ExecuteTest(GetParam());
  }
}

TEST_P(OtcheskovSLinearTopologyFuncTests, SeqTests) {
  ExecuteTest(GetParam());
}

const MessageData kData5 = {1, 2, 3, 4, 5};
const MessageData kData1 = {1};

const std::array<TestType, 4> kMpiParam2Proc = {
    {{{MessageHeader{.delivered = 0, .src = 0, .dest = 0, .data_size = 5}, kData5}, 1},
     {{MessageHeader{.delivered = 0, .src = 0, .dest = 1, .data_size = 5}, kData5}, 2},
     {{MessageHeader{.delivered = 0, .src = 1, .dest = 0, .data_size = 5}, kData5}, 3},
     {{MessageHeader{.delivered = 0, .src = 1, .dest = 1, .data_size = 5}, kData5}, 4}}};

const std::array<TestType, 6> kMpiParam4Proc = {
    {{{MessageHeader{.delivered = 0, .src = 0, .dest = 2, .data_size = 5}, kData5}, 1},
     {{MessageHeader{.delivered = 0, .src = 0, .dest = 3, .data_size = 5}, kData5}, 2},
     {{MessageHeader{.delivered = 0, .src = 1, .dest = 3, .data_size = 5}, kData5}, 3},
     {{MessageHeader{.delivered = 0, .src = 3, .dest = 0, .data_size = 5}, kData5}, 4},
     {{MessageHeader{.delivered = 0, .src = 2, .dest = 1, .data_size = 5}, kData5}, 5},
     {{MessageHeader{.delivered = 0, .src = 1, .dest = 2, .data_size = 5}, kData5}, 6}}};

const std::array<TestType, 2> kSeqParam = {
    {{{MessageHeader{.delivered = 0, .src = 0, .dest = 0, .data_size = 5}, kData5}, 1},
     {{MessageHeader{.delivered = 0, .src = 0, .dest = 0, .data_size = 1}, kData1}, 2}}};

const auto kMpiTasksList2Proc = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSLinearTopologyMPI, InType>(
    kMpiParam2Proc, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kMpiTasksList4Proc = std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSLinearTopologyMPI, InType>(
    kMpiParam4Proc, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kSeqTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<OtcheskovSLinearTopologySEQ, InType>(kSeqParam, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kMpiGtestValues2Proc = ppc::util::ExpandToValues(kMpiTasksList2Proc);
const auto kMpiGtestValues4Proc = ppc::util::ExpandToValues(kMpiTasksList4Proc);
const auto kSeqGtestValues = ppc::util::ExpandToValues(kSeqTasksList);

const auto kFuncTestName = OtcheskovSLinearTopologyFuncTests::PrintFuncTestName<OtcheskovSLinearTopologyFuncTests>;

INSTANTIATE_TEST_SUITE_P(Mpi2ProcsTests, OtcheskovSLinearTopologyMpi2ProcTests, kMpiGtestValues2Proc, kFuncTestName);
INSTANTIATE_TEST_SUITE_P(Mpi4ProcsTests, OtcheskovSLinearTopologyMpi4ProcTests, kMpiGtestValues4Proc, kFuncTestName);
INSTANTIATE_TEST_SUITE_P(SeqTests, OtcheskovSLinearTopologyFuncTests, kSeqGtestValues, kFuncTestName);

TEST_P(OtcheskovSLinearTopologyFuncTestsValidation, Validation) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kValidationTestParam = {{
    {{MessageHeader{.delivered = 0, .src = -1, .dest = 0, .data_size = 1}, kData1}, 1},
    {{MessageHeader{.delivered = 0, .src = 0, .dest = -1, .data_size = 1}, kData1}, 2},
    {{MessageHeader{.delivered = 0, .src = 0, .dest = 0, .data_size = 0}, MessageData{}}, 3},
    {{MessageHeader{.delivered = 1, .src = 0, .dest = 0, .data_size = 1}, kData1}, 4},
    {{MessageHeader{.delivered = 0, .src = 100000, .dest = 0, .data_size = 1}, kData1}, 5},
    {{MessageHeader{.delivered = 0, .src = 0, .dest = 100000, .data_size = 1}, kData1}, 6},
}};

const auto kValidationTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<OtcheskovSLinearTopologyMPI, InType>(
                       kValidationTestParam, PPC_SETTINGS_otcheskov_s_linear_topology),
                   ppc::util::AddFuncTask<OtcheskovSLinearTopologySEQ, InType>(
                       kValidationTestParam, PPC_SETTINGS_otcheskov_s_linear_topology));

const auto kValidationGtestValues = ppc::util::ExpandToValues(kValidationTestTasksList);

const auto kValidationFuncTestName =
    OtcheskovSLinearTopologyFuncTestsValidation::PrintFuncTestName<OtcheskovSLinearTopologyFuncTestsValidation>;

INSTANTIATE_TEST_SUITE_P(LinearTopologyTestsValidation, OtcheskovSLinearTopologyFuncTestsValidation,
                         kValidationGtestValues, kValidationFuncTestName);

}  // namespace
```

### 10.5. Приложение №5 — производительные тесты
Файл: `./tests/performance/main.cpp`.
```cpp
class OtcheskovSLinearTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 128000000;
  InType input_msg_;

  void SetUp() override {
    input_msg_.first = {.delivered = 0, .src = 0, .dest = 0, .data_size = 0};
    if (ppc::util::IsUnderMpirun()) {
      int proc_size{};
      MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
      input_msg_.first.dest = proc_size - 1;
    }

    input_msg_.second.resize(kDataSize);
    for (int i = 0; i < kDataSize; ++i) {
      input_msg_.second[static_cast<std::size_t>(i)] = i;
    }
  }

  bool CheckTestOutputData(OutType &output_msg) final {
    bool is_valid = false;
    const auto &[in_header, in_data] = input_msg_;
    const auto &[out_header, out_data] = output_msg;
    if (!ppc::util::IsUnderMpirun()) {
      is_valid = out_header.delivered != 0;
    } else {
      int proc_rank{};
      MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

      if (proc_rank == in_header.src || proc_rank == in_header.dest) {
        is_valid = (in_data == out_data) && out_header.delivered != 0;
      } else {
        is_valid = true;
      }
    }
    return is_valid;
  }

  InType GetTestInputData() final {
    return input_msg_;
  }
};

TEST_P(OtcheskovSLinearTopologyPerfTests, LinearTopologyPerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OtcheskovSLinearTopologyMPI>(PPC_SETTINGS_otcheskov_s_linear_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OtcheskovSLinearTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(LinearTopologyPerfTests, OtcheskovSLinearTopologyPerfTests, kGtestValues, kPerfTestName);
```