# pacman-1995-rl
Goal: train an RL agent to play the original pacman 1.0 (1995, Roar Thronaes, GPL-2+).
- game/ — the game's sources. Only minimal changes, each marked "// RL bridge". Do not touch the Windows (MSWIN) branches.
- Building the game: cd game && xmkmf && make. Needs build-essential, xutils-dev (xmkmf), libx11-dev, libncurses-dev. Build artifacts are in .gitignore.
- env/ — the gymnasium wrapper; talks to the game over a Unix socket.
- agents/, scripts/, configs/ — training. All hyperparameters only in configs/*.yaml.
- tools/ — helper utilities (demo client, prefix recording), data/ — recorded data (endgame prefixes), docs/ — results and logs, tests/ — pytest.
- docs/overnight.md — the experiment log: chronology, decisions, unclear points and "Ideas for later"; each session adds a dated section. docs/results.md — the results tables (evaluate.py rows and the summary of all runs).
- runs/ — runs and checkpoints, not in git (.gitignore); checkpoints are published only through GitHub releases.
- Python 3.10+, environment .venv. Run pytest before committing. Training longer than 10 minutes only in tmux.
- Public repository: github.com/AceEca-robo/pacman-1995-rl, branch main. Commits use AceEca-robo@users.noreply.github.com (set in the project's git config).
