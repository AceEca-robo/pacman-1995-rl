/* C port of the game, the same machine as sim/pacman_sim.py (see
   docs/sim_spec.md). A Sim is a plain struct: memcpy clones it. Built by
   sim/csim.py with gcc -O2 -shared; loaded with ctypes. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define W 33
#define H 23
#define CELLS (W * H)
#define LEVELS 16
#define FOODS 168
#define NCH 21                  /* env CHANNELS */

enum { UP, DOWN, LEFT, RIGHT, NONE, STILL };
enum { GATE, WALL, FOOD, SUPERFOOD, BLANK };
enum { RANDM, HUNTER, HUNTED, EATEN };
enum { DEAD, NORMAL, SUPER };
enum { NO_BONUS, BONUS_POINT, BONUS_LIFE };

typedef struct {
    int32_t x, y, sx, sy, gox, goy, st, d, deadtime, randmv;
} Ghost;

typedef struct {
    uint32_t r[31];
    int32_t ri;                 /* ring index of r[i - 31] */
    int32_t level, boardlevel, lives, eats, start_level;
    int64_t score, bonus;
    int32_t px, py, pst, pd, try_d, supertime;
    int32_t ghostseaten, bon, bon_life, d, phase;
    Ghost g[4];
    uint8_t cells[CELLS];
} Sim;

static uint8_t boards[LEVELS][CELLS];
static const int DX[6] = {0, 0, -1, 1, 0, 0};
static const int DY[6] = {-1, 1, 0, 0, 0, 0};
static const int LEFT_OF[4] = {LEFT, RIGHT, UP, DOWN};
static const int RIGHT_OF[4] = {RIGHT, LEFT, DOWN, UP};
static const int ACTION_DIR[5] = {UP, DOWN, LEFT, RIGHT, NONE};
static const int GSX[4] = {15, 17, 15, 17}, GSY[4] = {9, 9, 11, 11};

int sim_size(void) { return sizeof(Sim); }

void sim_set_board(int k, const uint8_t *cells) { memcpy(boards[k], cells, CELLS); }

/* --- glibc random() ----------------------------------------------------- */

static uint32_t rnd(Sim *s) {
    int k = s->ri;
    uint32_t v = s->r[k] + s->r[(k + 28) % 31];
    s->r[k] = v;
    s->ri = (k + 1) % 31;
    return v >> 1;
}

static void srnd(Sim *s, uint32_t seed) {
    int64_t r[34];
    int i;
    if (seed == 0) seed = 1;
    r[0] = (int32_t)seed;
    for (i = 1; i < 31; i++) {
        int64_t word = r[i - 1], hi = word / 127773, lo = word % 127773;
        word = 16807 * lo - 2836 * hi;
        if (word < 0) word += 2147483647;
        r[i] = word;
    }
    for (i = 31; i < 34; i++) r[i] = r[i - 31];
    for (i = 0; i < 3; i++) s->r[i] = (uint32_t)r[31 + i];
    for (i = 3; i < 31; i++) s->r[i] = (uint32_t)r[i];
    s->ri = 3;
    for (i = 0; i < 310; i++) rnd(s);
}

/* --- gamedata / board --------------------------------------------------- */

static void setboardlevel(Sim *s) {
    s->boardlevel = s->level > LEVELS ? 1 + (int)(rnd(s) % LEVELS) : s->level;
}

static void gd_start(Sim *s) {
    s->level = 1; s->lives = 3; s->score = 0; s->bonus = 5000; s->eats = FOODS;
    setboardlevel(s);
    if (s->start_level) { s->level = s->start_level; setboardlevel(s); }
}

static void pac_start(Sim *s) {
    s->px = 16; s->py = 17; s->pst = NORMAL; s->pd = STILL; s->try_d = RIGHT; s->supertime = 0;
}

static void ghost_start(Sim *s, Ghost *g) {
    g->x = g->sx; g->y = g->sy; g->st = RANDM; g->d = UP;
    s->ghostseaten = 0;
}

static void level_start(Sim *s) {
    int i;
    memcpy(s->cells, boards[s->boardlevel - 1], CELLS);
    pac_start(s);
    for (i = 0; i < 4; i++) ghost_start(s, &s->g[i]);
    s->d = STILL;
}

#define CELL(s, x, y) ((s)->cells[(y) * W + (x)])

/* --- pacman ------------------------------------------------------------- */

static void pac_go(Sim *s, int dd) {
    int xx, yy, w;
    if (dd != NONE) s->pd = s->try_d = dd;
    if (s->pst == DEAD) { s->pd = STILL; return; }
    if (s->pst == SUPER) {
        if (s->supertime) s->supertime--; else s->pst = NORMAL;
    }
    xx = s->px + DX[s->pd]; yy = s->py + DY[s->pd];
    w = CELL(s, xx, yy);
    if (w == WALL || w == GATE) {
        s->pd = STILL;
    } else if (w == FOOD) {
        s->px = xx; s->py = yy; CELL(s, xx, yy) = BLANK;
        if (s->eats) s->eats--;
        s->score += 10;
    } else if (w == SUPERFOOD) {
        s->px = xx; s->py = yy; CELL(s, xx, yy) = BLANK;
        s->pst = SUPER; s->supertime = 50;
    } else {
        s->px = xx; s->py = yy;
    }
}

/* --- ghosts ------------------------------------------------------------- */

static int wall(Sim *s, int x, int y) { return CELL(s, x, y) == WALL; }

static int otherdir(Sim *s, Ghost *g) {
    int a, b;
    if (g->d > RIGHT) return 1;
    a = LEFT_OF[g->d]; b = RIGHT_OF[g->d];
    return !wall(s, g->x + DX[a], g->y + DY[a]) || !wall(s, g->x + DX[b], g->y + DY[b]);
}

static void newdir(Sim *s, Ghost *g) {
    int a, b;
    if (g->d > RIGHT) { rnd(s); g->d = NONE; return; }
    a = LEFT_OF[g->d]; b = RIGHT_OF[g->d];
    if (wall(s, g->x + DX[a], g->y + DY[a]))
        g->d = wall(s, g->x + DX[b], g->y + DY[b]) ? NONE : b;
    else if (wall(s, g->x + DX[b], g->y + DY[b]))
        g->d = a;
    else
        g->d = (rnd(s) % 2) ? b : a;
}

static int searchpac(Sim *s, Ghost *g) {
    int px = s->px, py = s->py, x = g->x, y = g->y, dx, dy;
    if (px != x && py != y) return 0;
    dx = (px > x) - (px < x);
    dy = (py > y) - (py < y);
    while (!(x == px && y == py) && CELL(s, x, y) != WALL) { x += dx; y += dy; }
    if (!(x == px && y == py)) return 0;
    g->gox = px; g->goy = py;
    if (g->st == HUNTED) { dx = -dx; dy = -dy; }
    if (dx == 0 && dy == -1) g->d = UP;
    else if (dx == 0 && dy == 1) g->d = DOWN;
    else if (dx == 1 && dy == 0) g->d = RIGHT;
    else if (dx == -1 && dy == 0) g->d = LEFT;
    return 1;
}

static void lookforpac(Sim *s, Ghost *g) {
    if (s->pst == SUPER) { if (g->st != EATEN) g->st = HUNTED; }
    else if (g->st == HUNTED) { g->st = RANDM; s->ghostseaten = 0; }
    switch (g->st) {
    case HUNTER:
        if (searchpac(s, g) && g->x == s->px && g->y == s->py) g->d = STILL;
        break;
    case RANDM:
        if (searchpac(s, g)) g->st = HUNTER;
        break;
    case HUNTED:
        searchpac(s, g);
        break;
    }
}

static void ghost_go(Sim *s, Ghost *g) {
    int xx, yy;
    if (s->pst == SUPER) { if (g->st != EATEN) g->st = HUNTED; }
    else if (g->st == HUNTED) g->st = RANDM;
    switch (g->st) {
    case HUNTER:
        if (searchpac(s, g)) {
            if (g->x == s->px && g->y == s->py) g->d = STILL;
        } else if (g->gox == g->x && g->goy == g->y) {
            newdir(s, g);
            g->st = RANDM;
        }
        break;
    case RANDM:
        if (searchpac(s, g)) g->st = HUNTER;
        else if (otherdir(s, g) && rnd(s) % (uint32_t)g->randmv == 0) newdir(s, g);
        break;
    case HUNTED:
        searchpac(s, g);
        break;
    default:
        if (g->deadtime) g->deadtime--;
        else { g->st = RANDM; g->d = UP; }
    }
    xx = g->x + DX[g->d]; yy = g->y + DY[g->d];
    if (wall(s, xx, yy)) {
        newdir(s, g);
        xx = g->x + DX[g->d]; yy = g->y + DY[g->d];
    }
    g->x = xx; g->y = yy;
}

static void ghost_die(Sim *s, Ghost *g) {
    s->ghostseaten++;
    s->score += (int64_t)100 << (s->ghostseaten - 1);
    g->deadtime = 25; g->st = EATEN; g->x = g->sx; g->y = g->sy; g->d = STILL;
}

/* --- main loop ---------------------------------------------------------- */

static void check(Sim *s) {
    int i;
    if (s->bon != NO_BONUS && s->px == 16 && s->py == 15) {
        if (s->bon == BONUS_POINT) s->score += 1000; else s->lives++;
        s->bon = NO_BONUS;
    }
    for (i = 0; i < 4; i++) {
        Ghost *g = &s->g[i];
        if (g->x == s->px && g->y == s->py) {
            if (s->pst == SUPER) ghost_die(s, g);
            else { s->pst = DEAD; if (s->lives) s->lives--; break; }
        }
    }
}

static void tick(Sim *s) {
    int i;
    for (i = 0; i < 4; i++) lookforpac(s, &s->g[i]);
    pac_go(s, s->d);
    for (i = 0; i < 4; i++) ghost_go(s, &s->g[i]);
    check(s);
}

static void bonus_tick(Sim *s) {
    if (s->bon == NO_BONUS && (rnd(s) & 4095) == 0) {
        s->bon = (rnd(s) & 61440) ? BONUS_POINT : BONUS_LIFE;
        s->bon_life = 50;
    }
    if (s->bon != NO_BONUS && s->bon_life-- == 0) s->bon = NO_BONUS;
}

void sim_reset(Sim *s, uint32_t raw_seed, int level) {
    int i;
    memset(s, 0, sizeof(Sim));
    srnd(s, raw_seed);
    s->start_level = level;
    for (i = 0; i < 4; i++) {
        Ghost *g = &s->g[i];
        g->x = g->sx = GSX[i]; g->y = g->sy = GSY[i];
        g->st = RANDM; g->d = UP;
        g->randmv = (int)(rnd(s) % 8) + 4;
    }
    gd_start(s);
    level_start(s);
    tick(s);
    s->phase = 1;
}

void sim_advance(Sim *s, int action) {
    int i;
    s->d = ACTION_DIR[action];
    if (s->phase == 1) {
        bonus_tick(s);
        if (s->pst == SUPER) { pac_go(s, s->d); check(s); s->phase = 2; return; }
    }
    if (s->pst == DEAD) {
        s->bon = NO_BONUS;
        pac_start(s);
        for (i = 0; i < 4; i++) ghost_start(s, &s->g[i]);
    }
    if (s->bonus) s->bonus -= 5;
    if (!(s->lives && s->eats)) {
        if (s->eats) gd_start(s);
        else {
            s->score += s->bonus;
            s->level++; setboardlevel(s);
            s->bonus = 5000; s->eats = FOODS;
            if (!s->lives) gd_start(s);
        }
        level_start(s);
    }
    tick(s);
    s->phase = 1;
}

/* n sims at stride sizeof(Sim), one action each */
void sim_advance_many(Sim *s, int n, const int32_t *actions) {
    int i;
    for (i = 0; i < n; i++) sim_advance(&s[i], actions[i]);
}

/* --- JSON line (rlbridge.cc) -------------------------------------------- */

static const char DIRCHAR[] = "UDLRSS";
static const char CELLCHAR[] = "-#.o ";

static char dirchar(int d) { return d >= 0 && d < 6 ? DIRCHAR[d] : 'S'; }

int sim_json(const Sim *s, char *buf) {
    int n = 0, i, j;
    static const char *gst[4] = {"normal", "normal", "hunted", "eyes"};
    n += sprintf(buf + n, "{\"grid\":[");
    for (j = 0; j < H; j++) {
        buf[n++] = '"';
        for (i = 0; i < W; i++) buf[n++] = CELLCHAR[s->cells[j * W + i]];
        buf[n++] = '"';
        if (j < H - 1) buf[n++] = ',';
    }
    n += sprintf(buf + n, "],\"pacman\":{\"x\":%d,\"y\":%d,\"dir\":\"%c\",\"try_dir\":\"%c\"},\"ghosts\":[",
                 s->px, s->py, dirchar(s->pd), dirchar(s->try_d));
    for (i = 0; i < 4; i++)
        n += sprintf(buf + n, "%s{\"x\":%d,\"y\":%d,\"dir\":\"%c\",\"state\":\"%s\"}",
                     i ? "," : "", s->g[i].x, s->g[i].y, dirchar(s->g[i].d), gst[s->g[i].st]);
    n += sprintf(buf + n, "],\"bonus\":");
    if (s->bon != NO_BONUS)
        n += sprintf(buf + n, "{\"x\":16,\"y\":15,\"type\":\"%s\"}", s->bon == BONUS_POINT ? "point" : "life");
    else
        n += sprintf(buf + n, "null");
    n += sprintf(buf + n, ",\"score\":%lld,\"lives\":%d,\"level\":%d,\"supertime_left\":%d,\"done\":%s}\n",
                 (long long)s->score, s->lives, s->level, s->pst == SUPER ? s->supertime : 0,
                 s->lives ? "false" : "true");
    return n;
}

/* --- observation (env/pacman_env.py PacmanEnv._obs) --------------------- */

static void bfs(const uint8_t *cells, int block_gate, const int *src, int nsrc, int16_t *dist) {
    int q[CELLS], head = 0, tail = 0, i;
    for (i = 0; i < CELLS; i++) dist[i] = -1;
    for (i = 0; i < nsrc; i++)
        if (dist[src[i]] < 0) { dist[src[i]] = 0; q[tail++] = src[i]; }
    while (head < tail) {
        int c = q[head++], x = c % W, y = c / W, k;
        int nb[4] = {c - W, c + W, c - 1, c + 1};
        int ok[4] = {y > 0, y < H - 1, x > 0, x < W - 1};
        for (k = 0; k < 4; k++) {
            int m = nb[k];
            if (!ok[k] || dist[m] >= 0) continue;
            if (cells[m] == WALL || (block_gate && cells[m] == GATE)) continue;
            dist[m] = dist[c] + 1;
            q[tail++] = m;
        }
    }
}

/* grid: float32 (21 + cell_safety, H, W), zeroed here; vec: float32 (3,) */
void sim_obs(const Sim *s, float *grid, float *vec, int cell_safety) {
    int nch = NCH + (cell_safety ? 1 : 0), i, p = s->py * W + s->px;
    int supertime_left = s->pst == SUPER ? s->supertime : 0;
    int lives = s->lives < 9 ? s->lives : 9, level = s->level < LEVELS ? s->level : LEVELS;
    memset(grid, 0, sizeof(float) * nch * CELLS);
    for (i = 0; i < CELLS; i++) {
        int c = s->cells[i];
        if (c == WALL) grid[0 * CELLS + i] = 1;
        else if (c == GATE) grid[1 * CELLS + i] = 1;
        else if (c == FOOD) grid[2 * CELLS + i] = 1;
        else if (c == SUPERFOOD) grid[3 * CELLS + i] = 1;
    }
    grid[4 * CELLS + p] = 1;
    if (s->pd <= RIGHT) grid[(5 + s->pd) * CELLS + p] = 1;
    if (s->try_d <= RIGHT) grid[(9 + s->try_d) * CELLS + p] = 1;
    for (i = 0; i < 4; i++) {
        const Ghost *g = &s->g[i];
        int c = g->y * W + g->x, st = g->st == EATEN ? 15 : g->st == HUNTED ? 14 : 13;
        grid[st * CELLS + c] += 0.25f;
        if (g->d >= 0 && g->d <= RIGHT) grid[(16 + g->d) * CELLS + c] += 0.25f;
    }
    if (s->bon != NO_BONUS) grid[20 * CELLS + 15 * W + 16] = 1;
    if (cell_safety) {
        int16_t dp[CELLS], dg[CELLS];
        int src[4], ns = 0;
        float *out = grid + NCH * CELLS;
        bfs(s->cells, 1, &p, 1, dp);
        for (i = 0; i < 4; i++)
            if (s->g[i].st == RANDM || s->g[i].st == HUNTER) src[ns++] = s->g[i].y * W + s->g[i].x;
        if (supertime_left > 0 || ns == 0) {
            for (i = 0; i < CELLS; i++) out[i] = dp[i] >= 0;
        } else {
            bfs(s->cells, 0, src, ns, dg);
            for (i = 0; i < CELLS; i++) out[i] = dp[i] >= 0 && (dg[i] < 0 || dp[i] < dg[i]);
        }
    }
    vec[0] = (float)(lives / 3.0);
    vec[1] = (float)(supertime_left / 50.0);
    vec[2] = (float)(level / (double)LEVELS);
}

void sim_obs_many(const Sim *s, int n, float *grid, float *vec, int cell_safety) {
    int i, nch = NCH + (cell_safety ? 1 : 0);
    for (i = 0; i < n; i++) sim_obs(&s[i], grid + (size_t)i * nch * CELLS, vec + 3 * i, cell_safety);
}

/* summaries for planning: lives, level, done per sim */
void sim_info_many(const Sim *s, int n, int32_t *out) {
    int i;
    for (i = 0; i < n; i++) {
        out[3 * i] = s[i].lives; out[3 * i + 1] = s[i].level; out[3 * i + 2] = s[i].px + W * s[i].py;
    }
}
