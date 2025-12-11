#  Проверка лексикографической упорядоченности двух строк

- Студент: Тимофеев Никита Сергеевич, группа 3823Б1ПР2
- Технологии: SEQ, MPI
- Вариант: 26

## 1. Вступление
Применение технологии MPI и сравнение производительности программ с использованием MPI и последовательным выполнением, на примере задачи проверки лексикографической упорядоченности двух строк

## 2. Постановка задачи
Задача: на вход программы подаются две строки, необходимо определить лексикографическую упорядоченность каждой из них в отдельности.
Формат входных данных: ```std::pair<std::string, std::string>```
Формат выходных данных: ```std::pair<int, int>```
Необходимо создать две программы, одна из которых использует последовательное выполнение, другая - использует преимущества технологии MPI, затем сравнить их скорость.

## 3. Базовый алгоритм (Последовательный)
```cpp
bool TimofeevNLexicographicOrderingSEQ::RunImpl() {
  auto input = GetInput();

  // only true if comparison is true on every step
  for (size_t i = 0; !input.first.empty() && i < input.first.length() - 1; i++) {
    GetOutput().first &= static_cast<int>(input.first[i] <= input.first[i + 1]);
  }
  for (size_t i = 0; !input.second.empty() && i < input.second.length() - 1; i++) {
    GetOutput().second &= static_cast<int>(input.second[i] <= input.second[i + 1]);
  }

  return true;
}
```

## 4. Схема распараллеливания
- Для MPI: распараллеливание ведётся только относительно двух процессов. Сначала нулевой процесс получает все вхоные данные и посылает вторую строку первому для обработки, затем нулевой и первый процесс обрабатывают одним циклом for первую и вторую строку соответстсвенно, записывая редультаты в первый и второй элемент пары выходных данных соответственно, затем обмениваясь со всеми процессами результатами с помощью функции MPI_Bcast

## 5. Детали реализации
- Структура кода:
Есть классы ```TimofeevNLexicographicOrderingMPI``` и ```TimofeevNLexicographicOrderingSEQ```, там функция ```bool RunImpl()```. В функции для последовательного исполнения написан вышеприведённый код (раздел 3. Базовый алгоритм (Последовательный)). Код для функции параллельного исполнения приведён в Приложении 1.
- Реализация предполагает, что пустая строка является упорядоченной

## 6. Экспериментальная установка
- Аппаратное обеспечение / ОС: Intel(R) Core(TM) i7-3770, 8 ядер / 1 поток на ядро, 23.4 ГиБ ОЗУ, ОС: кубунту 24.04
- Инструментальная цепочка: gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0, clang-tidy-21, clang-format-21, тип сборки: Release
- Данные: тестовые данные для функциональных тестов находятся в файле tests/functional/main.cpp в виде списка типа ``` std::array<TestType, 7>```, тестовые данные для тестирования производительности представляют из себя пару строк, сплошь состоящих из символа 'a', и каждая из них длинной в 5e7 + 1 символ

## 7. Результаты и выводы

### 7.1 Корректность
Корректность была проверена с помощью функциональных тестов.

### 7.2 Производительность
Время предоставлено для запусков с 1, 2, 3 и 4 процессами соответственно
| Mode         | Time, s        | Speedup |
|--------------|----------------|---------|
| seq, task_run| 0.1220961571   | 1.00    |
| mpi, task_run| 0.1180528396   | 1.03    |
| seq, task_run| 0.1253298283   | 1.00    |
| mpi, task_run| 0.1004156796   | 1.24    |
| seq, task_run| 0.1354135513   | 1.00    |
| mpi, task_run| 0.1071434270   | 1.26    |
| seq, task_run| 0.1456462860   | 1.00    |
| mpi, task_run| 0.1094934970   | 1.33    |
Speedup для mpi = (seq time / mpi time)

## 8. Заключение
Были написаны две программы для сарвнения последовательного выполнения и mpi решения задачи проверки упорядоченности дву строк. В результате применения технологии MPI программа ускорилась.

## Приложение
Приложение 1. Код функции параллельного исполнения
```cpp
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
```