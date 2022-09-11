Ссылка на github репозиторий проекта: https://github.com/Frank-Way/sgm-pract1

Исходные данные для аппроксимации необходимо записать в файл "input.csv" и поместить в папку проекта. Формат файла:

    "x","y"
    <x1>,<y1>
    <x2>,<y2>
    ...
    <xn>,<yn>
Десятичный разделитель - ".", разделитель значений - "," (без пробелов!).

Выбор степени аппроксимирующего полинома выполняется указанием второго аргумента функции main:

    if __name__ == "__main__":
        is_plots_required = True  # нужно ли строить графики
        main_linear(is_plots_required, <степень полинома>)

Основная часть скрипта реализована на "чистом Python" и не требует дополнительных библиотек. Для построения графиков используется библиотека matplotlib. При отсутствии возможности установки библиотеки необходимо закомментировать или удалить следующие строки:

    from matplotlib import pyplot as plt
А также все строки с формированием графиков:

    if is_plots_required:  # формирование графика
        plt.plot(x, y, "o", label="исходные данные")
        ... оставшиеся строки с таким же отступом

После этого скрипт может быть запущен "голым" интерпретатором (например, с помощью интерпретатора на сайте replit.com, который позволяет также загружать и создавать различные файлы в проекте, в том числе csv-файлы).