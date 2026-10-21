"""All user-visible strings of the launcher (scripts/launcher.py) and of the
desktop entry, in English (the default), Russian and Ukrainian.

    t("ru", "watch")                    -> "Смотреть нейросеть"
    t("uk", "watch_hint", k=8)          -> "... 8 варіантів ..."

A value is either a string (str.format with the keyword arguments) or, for
words that follow a number, a function of the arguments.
"""

LANGS = ("en", "ru", "uk")
DEFAULT = "en"
LABELS = {"en": "EN", "ru": "RU", "uk": "UK"}


def plural(lang, n, forms):
    """forms: (one, few, many) for ru / uk, (one, other) for en."""
    if lang == "en":
        return forms[0] if n == 1 else forms[1]
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


STRINGS = {
    # --- menu ----------------------------------------------------------------
    "subtitle": {
        "en": "The original game by Roar Thronaes, and a neural network that learned to play it.",
        "ru": "Оригинальная игра Roar Thronaes и нейросеть, которая научилась в неё играть.",
        "uk": "Оригінальна гра Roar Thronaes і нейромережа, яка навчилася в неї грати.",
    },
    "play": {
        "en": "Play yourself",
        "ru": "Играть самому",
        "uk": "Грати самому",
    },
    "play_hint": {
        "en": "Arrow keys to move; hold Q to quit.",
        "ru": "Стрелки — движение; чтобы выйти, зажмите Q.",
        "uk": "Стрілки — рух; щоб вийти, затисніть Q.",
    },
    "watch": {
        "en": "Watch the neural network",
        "ru": "Смотреть нейросеть",
        "uk": "Дивитися нейромережу",
    },
    "watch_hint": {
        "en": lambda k: f"Before every move it plays out {k} possible futures, "
                        f"40 steps ahead, and avoids the deadly ones.",
        "ru": lambda k: f"Перед каждым ходом проигрывает {k} "
                        f"{plural('ru', k, ('вариант', 'варианта', 'вариантов'))} будущего "
                        f"на 40 шагов вперёд и обходит смертельные.",
        "uk": lambda k: f"Перед кожним ходом програє {k} "
                        f"{plural('uk', k, ('варіант', 'варіанти', 'варіантів'))} майбутнього "
                        f"на 40 кроків уперед і оминає смертельні.",
    },
    "dqn": {
        "en": "Network without the planner",
        "ru": "Нейросеть без планировщика",
        "uk": "Нейромережа без планувальника",
    },
    "dqn_hint": {
        "en": "The same network moving on first impression, with no look-ahead. "
              "Spot the difference.",
        "ru": "Та же сеть, но ходит сразу, по первому впечатлению, ничего не просчитывая "
              "наперёд. Почувствуйте разницу.",
        "uk": "Та сама мережа, але ходить одразу, за першим враженням, нічого не прораховуючи "
              "наперед. Відчуйте різницю.",
    },
    "level": {
        "en": "Level",
        "ru": "Уровень",
        "uk": "Рівень",
    },
    "level_hint": {
        "en": "Each of the 16 levels has its own maze.",
        "ru": "У каждого из 16 уровней свой лабиринт.",
        "uk": "Кожен із 16 рівнів має свій лабіринт.",
    },
    "smooth": {
        "en": "Smooth mode, K = 8",
        "ru": "Плавный режим, K = 8",
        "uk": "Плавний режим, K = 8",
    },
    "smooth_hint": {
        "en": "Fewer futures per move: quicker decisions, but it dies a bit more often.",
        "ru": "Меньше вариантов на ход: решает быстрее, но гибнет чуть чаще.",
        "uk": "Менше варіантів на хід: вирішує швидше, але гине трохи частіше.",
    },
    "cpu_note": {
        "en": "No NVIDIA GPU found: the network runs on the processor and may lag behind the game.",
        "ru": "Видеокарта NVIDIA не найдена: сеть считает на процессоре и может не поспевать "
              "за игрой.",
        "uk": "Відеокарту NVIDIA не знайдено: мережа рахує на процесорі й може не встигати "
              "за грою.",
    },
    "running": {
        "en": "The game is running in its own window. Close it to come back here.",
        "ru": "Игра идёт в отдельном окне. Закройте его, чтобы вернуться сюда.",
        "uk": "Гра йде в окремому вікні. Закрийте його, щоб повернутися сюди.",
    },
    "stop": {
        "en": "Stop",
        "ru": "Остановить",
        "uk": "Зупинити",
    },
    "last_game": {
        "en": "Last game: score {score}, level {level}.",
        "ru": "Прошлая игра: счёт {score}, уровень {level}.",
        "uk": "Попередня гра: рахунок {score}, рівень {level}.",
    },
    "version": {
        "en": "version {v}",
        "ru": "версия {v}",
        "uk": "версія {v}",
    },
    "license": {
        "en": "License: GPL-2.0-or-later",
        "ru": "Лицензия: GPL-2.0-or-later",
        "uk": "Ліцензія: GPL-2.0-or-later",
    },

    # --- errors --------------------------------------------------------------
    "err_no_x11": {
        "en": "No X11 display: DISPLAY is not set or the X server does not answer. "
              "The game only runs under X11 (or XWayland).",
        "ru": "Нет дисплея X11: переменная DISPLAY не задана или X-сервер не отвечает. "
              "Игра работает только под X11 (или XWayland).",
        "uk": "Немає дисплея X11: змінну DISPLAY не задано або X-сервер не відповідає. "
              "Гра працює лише під X11 (або XWayland).",
    },
    "err_no_game": {
        "en": "The game is not built yet. Run ./install.sh in the project folder.",
        "ru": "Игра ещё не собрана. Запустите ./install.sh в папке проекта.",
        "uk": "Гру ще не зібрано. Запустіть ./install.sh у теці проєкту.",
    },
    "err_no_checkpoint": {
        "en": "The network's weights (best_v04) are not here yet. Download them first.",
        "ru": "Весов нейросети (best_v04) пока нет. Сначала скачайте их.",
        "uk": "Ваг нейромережі (best_v04) поки немає. Спершу завантажте їх.",
    },
    "err_no_tool": {
        "en": "Missing programs: {tools}. Install them with: sudo apt install {pkgs}",
        "ru": "Не хватает программ: {tools}. Установите их: sudo apt install {pkgs}",
        "uk": "Бракує програм: {tools}. Встановіть їх: sudo apt install {pkgs}",
    },
    "err_no_module": {
        "en": "The Python package {name} is not installed. Run ./install.sh again.",
        "ru": "Не установлен Python-пакет {name}. Запустите ./install.sh ещё раз.",
        "uk": "Не встановлено Python-пакет {name}. Запустіть ./install.sh ще раз.",
    },
    "err_game_start": {
        "en": "The game did not start. Rebuild it with ./install.sh. Details: {log}",
        "ru": "Игра не запустилась. Пересоберите её через ./install.sh. Подробности: {log}",
        "uk": "Гра не запустилася. Перезберіть її через ./install.sh. Подробиці: {log}",
    },
    "err_desync": {
        "en": "The planner lost track of the game. This should never happen; "
              "please report it and attach {log}",
        "ru": "Планировщик потерял синхронизацию с игрой. Такого быть не должно — "
              "сообщите нам, пожалуйста, и приложите {log}",
        "uk": "Планувальник втратив синхронізацію з грою. Такого не має бути — "
              "повідомте нам, будь ласка, і додайте {log}",
    },
    "err_gpu_memory": {
        "en": "The graphics card ran out of memory. Close other programs that use it, "
              "or turn on smooth mode.",
        "ru": "Видеокарте не хватило памяти. Закройте другие программы, которые её занимают, "
              "или включите плавный режим.",
        "uk": "Відеокарті забракло пам'яті. Закрийте інші програми, що її займають, "
              "або увімкніть плавний режим.",
    },
    "err_crash": {
        "en": "Something went wrong (exit code {code}). Details: {log}",
        "ru": "Что-то пошло не так (код выхода {code}). Подробности: {log}",
        "uk": "Щось пішло не так (код виходу {code}). Подробиці: {log}",
    },
    "err_launch": {
        "en": "Could not start it: {error}",
        "ru": "Не удалось запустить: {error}",
        "uk": "Не вдалося запустити: {error}",
    },

    # --- checkpoint download --------------------------------------------------
    "dl_title": {
        "en": "Download the neural network?",
        "ru": "Скачать нейросеть?",
        "uk": "Завантажити нейромережу?",
    },
    "dl_text": {
        "en": "To watch the network play, the launcher needs its weights: one file of "
              "{size} MB from the project's release on GitHub. It will be saved to {path} "
              "and checked against its SHA-256 checksum.",
        "ru": "Чтобы посмотреть, как играет нейросеть, нужны её веса — один файл на {size} МБ "
              "из релиза проекта на GitHub. Он сохранится в {path}, а его контрольная сумма "
              "SHA-256 будет проверена.",
        "uk": "Щоб подивитися, як грає нейромережа, потрібні її ваги — один файл на {size} МБ "
              "із релізу проєкту на GitHub. Його буде збережено в {path} і звірено "
              "контрольну суму SHA-256.",
    },
    "dl_yes": {
        "en": "Download",
        "ru": "Скачать",
        "uk": "Завантажити",
    },
    "dl_no": {
        "en": "Not now",
        "ru": "Не сейчас",
        "uk": "Не зараз",
    },
    "dl_cancel": {
        "en": "Cancel",
        "ru": "Отмена",
        "uk": "Скасувати",
    },
    "dl_retry": {
        "en": "Try again",
        "ru": "Ещё раз",
        "uk": "Ще раз",
    },
    "dl_progress": {
        "en": "{done} of {total} MB",
        "ru": "{done} из {total} МБ",
        "uk": "{done} з {total} МБ",
    },
    "dl_verifying": {
        "en": "Checking the checksum…",
        "ru": "Проверяю контрольную сумму…",
        "uk": "Перевіряю контрольну суму…",
    },
    "dl_done": {
        "en": "Done: the network is in place.",
        "ru": "Готово: нейросеть на месте.",
        "uk": "Готово: нейромережа на місці.",
    },
    "dl_err_net": {
        "en": "The download failed ({error}). Check the internet connection and try again.",
        "ru": "Скачать не получилось ({error}). Проверьте подключение к интернету "
              "и попробуйте ещё раз.",
        "uk": "Завантажити не вдалося ({error}). Перевірте підключення до інтернету "
              "й спробуйте ще раз.",
    },
    "dl_err_hash": {
        "en": "The file arrived damaged (the checksum does not match) and was deleted. "
              "Try again.",
        "ru": "Файл пришёл повреждённым (контрольная сумма не совпала) и удалён. "
              "Попробуйте ещё раз.",
        "uk": "Файл прийшов пошкодженим (контрольна сума не збіглася), його видалено. "
              "Спробуйте ще раз.",
    },

    # --- desktop entry (install.sh) --------------------------------------------
    "desktop_name": {
        "en": "Pacman 1995 RL",
        "ru": "Pacman 1995 и нейросеть",
        "uk": "Pacman 1995 і нейромережа",
    },
    "desktop_comment": {
        "en": "Play the original 1995 pacman or watch a neural network play it",
        "ru": "Сыграйте в оригинальный pacman 1995 года или посмотрите, как в него играет "
              "нейросеть",
        "uk": "Зіграйте в оригінальний pacman 1995 року або подивіться, як у нього грає "
              "нейромережа",
    },
}


def t(lang, key, **kw):
    entry = STRINGS[key]
    s = entry.get(lang, entry[DEFAULT])
    return s(**kw) if callable(s) else s.format(**kw)
