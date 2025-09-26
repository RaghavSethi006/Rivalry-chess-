# Phase 0 — Scope, Stack, and Repo (½ day)

**Goal:** lock decisions so you don’t churn later.

* **Engine style:** classical search (minimax + alpha-beta) at depth 2–3 plies.
* **Evaluation:** ML predictor (logistic regression → gradient boosting) that outputs **P(win for side-to-move)**.
* **Move generation:** use `python-chess` for legality & FEN/PGN handling (fast, avoids bugs).
  *(If you insist on “from-scratch” later, swap in your own generator as a stretch goal.)*
* **Backend:** **FastAPI** (typed, auto-docs), `uvicorn`.
* **ML:** scikit-learn, pandas, numpy, joblib.
* **Frontend:** **React + Vite**, `react-chessboard`, `chess.js` (client-side legality/help), Tailwind for UI.
* **Storage:** start with files (`models/`, `data/processed/`), later SQLite if needed.
* **Rating goal:** play reasonably at **\~700–900 Elo** with depth 2–3 + simple pruning & ML eval.
* **Repo layout:**

  ```
  smart-chess/
  ├─ engine/
  │  ├─ search.py           # minimax + alpha-beta + move ordering
  │  ├─ eval_ml.py          # model loading + evaluation wrapper
  │  ├─ features.py         # board → features
  │  ├─ utils.py            # FEN helpers, timers, etc.
  ├─ ml/
  │  ├─ pgn_to_rows.py      # PGN → (FEN, result, meta)
  │  ├─ make_dataset.py     # sampling positions, split, save
  │  ├─ train.py            # train + calibrate + export model.joblib
  │  ├─ metrics.py          # ROC/AUC, Brier, calibration curves
  ├─ api/
  │  ├─ main.py             # FastAPI endpoints
  │  ├─ schemas.py          # Pydantic request/response models
  ├─ web/                   # React app
  ├─ data/
  │  ├─ raw/                # PGNs
  │  ├─ processed/          # parquet/csv of positions
  ├─ models/                # model.joblib, model_meta.json
  ├─ tests/                 # pytest unit + e2e
  ├─ scripts/               # small CLIs (bench, selfplay)
  └─ README.md
  ```

---

# Phase 1 — Minimal Playable Engine (2–3 days)

**Goal:** a bot that plays from a FEN with shallow search and a dumb heuristic (no ML yet).

1. **Board I/O**

   * Accept **FEN**; generate legal moves with `python-chess`.
   * Return moves in **UCI** (e.g., `e2e4`).

2. **Baseline evaluation** (temporary):

   * Material count (P=1, N=3, B=3.2, R=5, Q=9).
   * Small bonuses: development (minor piece on c/d/e/f ranks), castling done, central pawns.
   * Mobility: `0.05 * (my_legal_moves - opp_legal_moves)`.

3. **Search**

   * **Minimax with alpha-beta**, fixed depth = 2 (later 3).
   * **Move ordering:** captures first (MVV-LVA), then checks, then others.
   * **Time control:** simple per-move limit (e.g., 200–400 ms) or fixed depth.

4. **CLI smoke test**

   * Play as Black/White from start FEN.
   * Ensure legality, no crashes, returns within time.

**Acceptance:** Engine responds < 0.5s/move at depth 2, never plays illegal moves.

---

# Phase 2 — Data & Features for ML (2–3 days)

**Goal:** dataset of positions → features → label.

1. **Data ingestion**

   * Load **PGN** games (e.g., lichess dumps you download locally).
   * For each game, **sample positions** every N plies (e.g., every 2 plies) to avoid correlation.
   * Store rows: `{fen, side_to_move, result (1/0/0.5), ply, rated?, avg_elo}`.

2. **Labeling**

   * **Target = game result from the position’s perspective**:

     * If side-to-move is White: label = result (1, 0.5, 0).
     * If side-to-move is Black: label = 1-result.

3. **Features** (keep it tabular & fast)

   * **Material**: counts & total material diff.
   * **King safety**: castled?, pawn shield (pawns on f/g/h or c/d/e depending on side), open files near king.
   * **Mobility**: legal move counts for side to move (compute once).
   * **Structure**: doubled, isolated, passed pawns counts.
   * **Piece-square tables**: sum(PST\[pieces]) for each piece type (simple 64-length arrays).
   * **Game/meta**: ply number (opening/middlegame/endgame proxy).

4. **Splits**

   * Train/val/test by **game**, not by position (avoid leakage).
   * Balance: undersample huge openings; ensure diversity.

**Acceptance:** A `data/processed/positions.parquet` with ≥100k rows and 40–120 features.

---

# Phase 3 — Train the Predictor (2–3 days)

**Goal:** a calibrated `predict_win(fen)` model.

1. **Baselines**

   * **Logistic Regression** (with regularization).
   * **GradientBoostingClassifier** (or HistGradientBoosting).

2. **Metrics**

   * AUC, **Brier score**, log loss, and **calibration curve**.
   * Track performance by **phase** (opening/middle/end) using ply bins.

3. **Calibration**

   * If tree model, add **isotonic** or **Platt scaling** on validation set.

4. **Export**

   * Save `model.joblib` and `model_meta.json` (feature list, version, training stats).

**Acceptance:** Model produces sensible probabilities (e.g., start FEN \~0.5), passes calibration sanity.

---

# Phase 4 — Hybrid Engine (Search + ML Eval) (2–3 days)

**Goal:** replace heuristic eval with ML probability.

1. **Eval glue**

   * At search leaf nodes, call `predict_win(features(fen))`.
   * Convert probability p to a **search score** (for minimax):

     * Option A: work directly with **expected value** in \[0,1], negate for opponent.
     * Option B: map to centipawn-like score: `score = log(p/(1-p)) * K` (K≈100).

2. **Practical speed**

   * **Cache features** per FEN (LRU dict).
   * Batch evaluate leaf nodes when possible (optional micro-batching).

3. **Blunder checks**

   * Always check for **mate in 1** (tactical sanity) by scanning checks/captures quickly.

4. **Parameters to expose**

   * `depth`: 2 or 3.
   * `use_quiescence`: off by default; optional captures-only extension depth +1.
   * `time_ms`: per move cap.

**Acceptance:** Plays stronger than Phase 1 vs yourself; fewer one-move blunders; still <1s/move.

---

# Phase 5 — API (FastAPI) (1–2 days)

**Goal:** clean HTTP interface to the engine.

* **Endpoints**

  * `POST /move/best`

    * **Req**: `{ fen: string, depth?: int=2, time_ms?: int=400, ml?: bool=true }`
    * **Res**: `{ bestMove: "e2e4", pv: ["e2e4","e7e5",...], score: 0.18, nodes: 12345, duration_ms: 312 }`
  * `POST /predict`

    * **Req**: `{ fen: string }`
    * **Res**: `{ p_win: 0.54, features_used: {...} }`
  * `GET /health` → `{status:"ok", model_version:"…"}`
* **Infra**

  * Load `model.joblib` once at startup.
  * CORS allowlist for your frontend.
  * Simple logging & request timeouts.

**Acceptance:** Can curl a FEN and get a legal best move and a probability.

---

# Phase 6 — Frontend (React) (2–3 days)

**Goal:** playable web app.

* **Pages/Components**

  * `Board` (react-chessboard).
  * Right pane: **Win prob** display, **engine move log**, PV line.
  * Controls: **Play as White/Black**, **Depth slider (2–3)**, **Time/Move**, **New game**, **Undo** (client-side only), **Hint**.
* **Flow**

  * User moves → send current FEN to `/move/best` → animate engine move.
  * Show **live win probability** from `/predict`.
* **Polish**

  * Loading spinners, resign/draw buttons.
  * Simple theming with Tailwind.

**Acceptance:** Smooth play vs bot, engine responds within your time cap.

---

# Phase 7 — Evaluation & Tuning (2–3 days)

**Goal:** hit the \~700–900 Elo feel.

* **Self-play ladders:** depth-2 vs depth-3; ML-on vs ML-off (baseline); record results.
* **A/B heuristics:** toggle mobility weight, PSTs in features, quiescence on captures only.
* **Time control knobs:** set `time_ms` \~200–500 ms → feels human-intermediate.
* **Opening book (optional):** a tiny JSON of 20–40 common book lines to avoid silly openings.
* **Tactical sanity:** always run a quick **checkmate/capture scan** before trusting ML score.

**Acceptance:** Your bot avoids blatant blunders, trades reasonably, mates in simple positions, and feels \~700–900 Elo in casual blitz games.

---

# Phase 8 — Deploy (1–2 days)

**Goal:** shareable app.

* **Containerize** with Docker (backend + model).
* **Serve** with Uvicorn/Gunicorn.
* **Host**: Railway/Render/Fly.io (free tiers), or a VPS.
* **Frontend**: static hosting (Vercel/Netlify) with API URL env var.
* **Docs**: README with screenshots, how-to, and a brief “Design Notes”.

---

# Stretch Goals (nice-to-have)

* **Explain move:** show top 3 features that swung the eval (LR coefficients or tree feature importances).
* **Difficulty levels:** depth, time cap, and “blunder rate” (randomly choose a sub-optimal move with small prob).
* **Opening book from your games:** learn a micro-book from your own PGNs.
* **Endgame table rule-of-thumbs:** simple K+P vs K heuristics (push passed pawns, opposition).

---

## Milestone Plan (example 3–4 weeks)

* **Week 1:** Phases 1–2 (engine skeleton + dataset)
* **Week 2:** Phases 3–4 (train model + hybrid eval)
* **Week 3:** Phases 5–6 (API + React app)
* **Week 4:** Phases 7–8 (tuning + deploy) and stretch goals
