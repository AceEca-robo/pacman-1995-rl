# pacman-rl
Цель: обучить RL-агента играть в оригинальную игру pacman 1.0 (1995, Roar Thronaes, GPL-2+).
- game/ — исходники игры. Правки только минимальные, каждая помечена "// RL bridge". Windows-ветки (MSWIN) не трогать.
- Сборка игры: cd game && xmkmf && make. Нужны build-essential, xutils-dev (xmkmf), libx11-dev, libncurses-dev. Артефакты сборки в .gitignore.
- env/ — gymnasium-обёртка, общается с игрой через Unix-сокет.
- agents/, scripts/, configs/ — обучение. Все гиперпараметры только в configs/*.yaml.
- tools/ — вспомогательные утилиты (демо-клиент, запись префиксов), data/ — записанные данные (префиксы концовок), docs/ — результаты и журналы, tests/ — pytest.
- runs/ — прогоны и чекпоинты, в git не попадают (.gitignore); чекпоинты для публикации — только через GitHub release.
- Python 3.10+, окружение .venv. Перед коммитом: pytest. Обучение дольше 10 минут — только в tmux.
- Публичный репозиторий: github.com/AceEca-robo/pacman-1995-rl, ветка main. Коммиты — с адресом AceEca-robo@users.noreply.github.com (стоит в git config проекта).
