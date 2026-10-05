# pacman-rl
Цель: обучить RL-агента играть в оригинальную игру pacman 1.0 (1995, Roar Thronaes, GPL-2+).
- game/ — исходники игры. Правки только минимальные, каждая помечена "// RL bridge". Windows-ветки (MSWIN) не трогать.
- Сборка игры: cd game && xmkmf && make. Нужен libncurses-dev. Артефакты сборки в .gitignore.
- env/ — gymnasium-обёртка, общается с игрой через Unix-сокет.
- agents/, scripts/, configs/ — обучение. Все гиперпараметры только в configs/*.yaml.
- Python 3.10+, окружение .venv. Перед коммитом: pytest. Обучение дольше 10 минут — только в tmux.
