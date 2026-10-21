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
    "cpu_note": {
        "en": "No NVIDIA GPU found: the network thinks on the processor, so the game runs "
              "in slow motion. Smooth mode is about four times faster.",
        "ru": "Видеокарта NVIDIA не найдена: сеть думает на процессоре, и игра идёт как "
              "в замедленной съёмке. В плавном режиме примерно вчетверо быстрее.",
        "uk": "Відеокарту NVIDIA не знайдено: мережа думає на процесорі, і гра йде як "
              "у сповільненій зйомці. У плавному режимі приблизно вчетверо швидше.",
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

    "play_hint": {
        "en": "Arrow keys to move, P to pause, Esc back to the menu.",
        "ru": "Стрелки — движение, P — пауза, Esc — назад в меню.",
        "uk": "Стрілки — рух, P — пауза, Esc — назад до меню.",
    },
    "human_net": {
        "en": "Show what the network sees",
        "ru": "Показывать, что видит сеть",
        "uk": "Показувати, що бачить мережа",
    },
    "human_net_short": {
        "en": "it thinks along, you steer",
        "ru": "она думает, ведёте вы",
        "uk": "вона думає, керуєте ви",
    },
    "smooth_short": {
        "en": "quicker, dies a bit more often",
        "ru": "быстрее, но гибнет чуть чаще",
        "uk": "швидше, але гине трохи частіше",
    },
    "keys_title": {
        "en": "Keys",
        "ru": "Клавиши",
        "uk": "Клавіші",
    },
    "keys_list": {
        "en": "Esc — back to the menu; in the menu: quit|P — pause|F11 — full screen"
              "|N — in your game: show or hide the network",
        "ru": "Esc — назад в меню, а в меню — выход|P — пауза|F11 — во весь экран"
              "|N — в вашей игре: показать или скрыть сеть",
        "uk": "Esc — назад до меню, а в меню — вихід|P — пауза|F11 — на весь екран"
              "|N — у вашій грі: показати або сховати мережу",
    },
    "starting": {
        "en": "Starting the game…",
        "ru": "Запускаю игру…",
        "uk": "Запускаю гру…",
    },
    "status_keys": {
        "en": "P — pause    Esc — menu    F11 — full screen",
        "ru": "P — пауза    Esc — меню    F11 — во весь экран",
        "uk": "P — пауза    Esc — меню    F11 — на весь екран",
    },
    "status_keys_human": {
        "en": "N — network view    P — pause    Esc — menu",
        "ru": "N — взгляд сети    P — пауза    Esc — меню",
        "uk": "N — погляд мережі    P — пауза    Esc — меню",
    },

    # --- pause, game over -------------------------------------------------------
    "paused": {
        "en": "Paused",
        "ru": "Пауза",
        "uk": "Пауза",
    },
    "pause_continue": {
        "en": "Continue",
        "ru": "Продолжить",
        "uk": "Продовжити",
    },
    "pause_menu": {
        "en": "Back to the menu",
        "ru": "В меню",
        "uk": "До меню",
    },
    "pause_quit": {
        "en": "Quit",
        "ru": "Выйти",
        "uk": "Вийти",
    },
    "game_over": {
        "en": "Game over: score {score}, level {level}",
        "ru": "Игра окончена: счёт {score}, уровень {level}",
        "uk": "Гру закінчено: рахунок {score}, рівень {level}",
    },
    "game_over_keys": {
        "en": "Enter — play again    Esc — menu",
        "ru": "Enter — ещё раз    Esc — в меню",
        "uk": "Enter — ще раз    Esc — до меню",
    },

    # --- your game: the panels instead of the network's ---------------------------
    "how_title": {
        "en": "HOW TO PLAY",
        "ru": "КАК ИГРАТЬ",
        "uk": "ЯК ГРАТИ",
    },
    "how_text": {
        "en": "Eat every dot to clear the maze; each of the 16 levels has a maze of its own."
              "|A ghost that touches you takes a life. The big dots turn the ghosts blue "
              "for a while: then you can eat them."
              "|Now and then a bonus appears: points or an extra life.",
        "ru": "Съешьте все точки, чтобы пройти лабиринт; у каждого из 16 уровней он свой."
              "|Призрак, коснувшись вас, отнимает жизнь. Большие точки ненадолго делают "
              "призраков синими — тогда их можно съесть."
              "|Иногда появляется бонус: очки или лишняя жизнь.",
        "uk": "З'їжте всі точки, щоб пройти лабіринт; у кожного з 16 рівнів він свій."
              "|Привид, торкнувшись вас, забирає життя. Великі точки ненадовго роблять "
              "привидів синіми — тоді їх можна з'їсти."
              "|Іноді з'являється бонус: очки або додаткове життя.",
    },
    "controls_title": {
        "en": "CONTROLS",
        "ru": "УПРАВЛЕНИЕ",
        "uk": "КЕРУВАННЯ",
    },
    "ctl_arrows": {
        "en": "hold an arrow: pacman turns where the maze lets it",
        "ru": "держите стрелку: пакман повернёт, где позволит лабиринт",
        "uk": "тримайте стрілку: пакман поверне, де дозволить лабіринт",
    },
    "ctl_n": {
        "en": "show or hide what the network sees",
        "ru": "показать или скрыть, что видит сеть",
        "uk": "показати або сховати, що бачить мережа",
    },
    "ctl_p": {
        "en": "pause",
        "ru": "пауза",
        "uk": "пауза",
    },
    "ctl_esc": {
        "en": "back to the menu",
        "ru": "назад в меню",
        "uk": "назад до меню",
    },
    "ctl_f11": {
        "en": "full screen",
        "ru": "во весь экран",
        "uk": "на весь екран",
    },
    "game_title": {
        "en": "YOUR GAME",
        "ru": "ВАША ИГРА",
        "uk": "ВАША ГРА",
    },
    "lbl_score": {
        "en": "score",
        "ru": "счёт",
        "uk": "рахунок",
    },
    "lbl_lives": {
        "en": "lives",
        "ru": "жизни",
        "uk": "життя",
    },
    "lbl_level": {
        "en": "level",
        "ru": "уровень",
        "uk": "рівень",
    },
    "lbl_maze": {
        "en": "maze",
        "ru": "лабиринт",
        "uk": "лабіринт",
    },
    "rec_title": {
        "en": "THIS SESSION",
        "ru": "ЗА ЭТОТ ЗАПУСК",
        "uk": "ЗА ЦЕЙ ЗАПУСК",
    },
    "rec_games": {
        "en": "games played",
        "ru": "сыграно игр",
        "uk": "зіграно ігор",
    },
    "rec_best": {
        "en": "best score",
        "ru": "лучший счёт",
        "uk": "найкращий рахунок",
    },
    "rec_level": {
        "en": "furthest level",
        "ru": "дальше всего — уровень",
        "uk": "найдалі — рівень",
    },
    "net_hint": {
        "en": "Press N to see your game through the network's eyes: what it notices "
              "and which way it would go.",
        "ru": "Нажмите N, чтобы увидеть свою игру глазами сети: что она замечает "
              "и куда пошла бы сама.",
        "uk": "Натисніть N, щоб побачити свою гру очима мережі: що вона помічає "
              "і куди пішла б сама.",
    },
    "net_missing": {
        "en": "The network's weights are not downloaded yet: the menu offers them.",
        "ru": "Веса сети ещё не скачаны — меню предложит их скачать.",
        "uk": "Ваги мережі ще не завантажено — меню запропонує їх завантажити.",
    },

    # --- the network view (scripts/netviz.py) -------------------------------------
    "viz_caption_net": {
        "en": "pacman 1.0 (1995) — played by the network",
        "ru": "pacman 1.0 (1995) — играет нейросеть",
        "uk": "pacman 1.0 (1995) — грає нейромережа",
    },
    "viz_caption_sim": {
        "en": "pacman 1.0 (1995) — played by the network (sim)",
        "ru": "pacman 1.0 (1995) — играет нейросеть (симулятор)",
        "uk": "pacman 1.0 (1995) — грає нейромережа (симулятор)",
    },
    "viz_caption_human": {
        "en": "pacman 1.0 (1995) — your game",
        "ru": "pacman 1.0 (1995) — играете вы",
        "uk": "pacman 1.0 (1995) — граєте ви",
    },
    "viz_input": {
        "en": "1  INPUT",
        "ru": "1  ВХОД",
        "uk": "1  ВХІД",
    },
    "viz_walls": {
        "en": "walls + food",
        "ru": "стены и еда",
        "uk": "стіни та їжа",
    },
    "viz_ghosts": {
        "en": "ghosts: normal / hunted / eyes",
        "ru": "призраки: обычные / синие / глаза",
        "uk": "привиди: звичайні / сині / очі",
    },
    "viz_safety": {
        "en": "cell safety (input)",
        "ru": "безопасные клетки (вход)",
        "uk": "безпечні клітинки (вхід)",
    },
    "viz_exits": {
        "en": "safe exits (derived)",
        "ru": "безопасные выходы (вычислено)",
        "uk": "безпечні виходи (обчислено)",
    },
    "viz_acts": {
        "en": "2  ACTIVATIONS (top 16 maps per conv layer)",
        "ru": "2  АКТИВАЦИИ (16 самых сильных карт слоя)",
        "uk": "2  АКТИВАЦІЇ (16 найсильніших карт шару)",
    },
    "viz_saliency": {
        "en": "3  SALIENCY  |dQ({a})/d input|",
        "ru": "3  ЗНАЧИМОСТЬ  |dQ({a})/d вход|",
        "uk": "3  ЗНАЧУЩІСТЬ  |dQ({a})/d вхід|",
    },
    "viz_output": {
        "en": "4  OUTPUT  Q per action",
        "ru": "4  ВЫХОД  Q каждого хода",
        "uk": "4  ВИХІД  Q кожного ходу",
    },
    "viz_output_human": {
        "en": "4  WHERE THE NETWORK WOULD GO",
        "ru": "4  КУДА ПОШЛА БЫ СЕТЬ",
        "uk": "4  КУДИ ПІШЛА Б МЕРЕЖА",
    },
    "viz_greedy": {
        "en": "greedy",
        "ru": "лучший",
        "uk": "найкращий",
    },
    "viz_you": {
        "en": "you",
        "ru": "вы",
        "uk": "ви",
    },
    "viz_dueling": {
        "en": "dueling heads",
        "ru": "две головы (dueling)",
        "uk": "дві голови (dueling)",
    },
    "viz_planner": {
        "en": "5  PLANNER  K={k} rollouts x H={h} steps per direction",
        "ru": "5  ПЛАНИРОВЩИК  K={k} прогонов по H={h} шагов на направление",
        "uk": "5  ПЛАНУВАЛЬНИК  K={k} прогонів по H={h} кроків на напрямок",
    },
    "viz_planner_off": {
        "en": "5  PLANNER  (off: the network plays alone)",
        "ru": "5  ПЛАНИРОВЩИК  (выключен: сеть играет сама)",
        "uk": "5  ПЛАНУВАЛЬНИК  (вимкнено: мережа грає сама)",
    },
    "viz_death_share": {
        "en": "death share",
        "ru": "доля гибелей",
        "uk": "частка загибелей",
    },
    "viz_drop": {
        "en": "drop if > {t}",
        "ru": "отбросить при > {t}",
        "uk": "відкинути при > {t}",
    },
    "viz_shown_only": {
        "en": "shown only",
        "ru": "только для показа",
        "uk": "лише для показу",
    },
    "viz_red_lost": {
        "en": "red: lost a life",
        "ru": "красным: потеря жизни",
        "uk": "червоним: втрата життя",
    },
    "viz_status": {
        "en": "tick {tick}   level {level} → maze {maze}   lives {lives}   deaths {deaths}   "
              "score {score}",
        "ru": "ход {tick}   уровень {level} → лабиринт {maze}   жизни {lives}   "
              "гибели {deaths}   счёт {score}",
        "uk": "хід {tick}   рівень {level} → лабіринт {maze}   життя {lives}   "
              "загибелі {deaths}   рахунок {score}",
    },
    "viz_ms_decision": {
        "en": "{ms} ms / decision",
        "ru": "{ms} мс на решение",
        "uk": "{ms} мс на рішення",
    },
    "viz_ms_drawing": {
        "en": "(+{ms} ms drawing)",
        "ru": "(+{ms} мс отрисовка)",
        "uk": "(+{ms} мс малювання)",
    },

    # --- errors --------------------------------------------------------------
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
        "en": "Something went wrong: {error}. Details: {log}",
        "ru": "Что-то пошло не так: {error}. Подробности: {log}",
        "uk": "Щось пішло не так: {error}. Подробиці: {log}",
    },
    "err_no_window": {
        "en": "Could not open a window: no graphical session (neither X11 nor Wayland) is running.",
        "ru": "Не удалось открыть окно: графический сеанс (X11 или Wayland) не запущен.",
        "uk": "Не вдалося відкрити вікно: графічний сеанс (X11 або Wayland) не запущено.",
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
