# Engine Contracts

These identifiers define stable observable requirements. Tests should cite an identifier when a test exists specifically to protect one of these contracts.

- **CORE-01** — `board_make` followed by `board_unmake` restores the complete logical position.
- **CORE-02** — Mailbox, bitboards, and cached occupancies describe the same pieces.
- **CORE-03** — The incremental Zobrist key equals `board_compute_hash` for the current state.
- **CORE-04** — Legal move generation matches reference perft values.
- **UCI-01** — The engine completes the UCI handshake and readiness protocol.
- **UCI-02** — `stop` terminates an active infinite search and produces `bestmove`.
- **UCI-03** — Documented UCI options remain present unless the contract is deliberately revised.
- **NNUE-01** — Incremental NNUE evaluation equals a full accumulator rebuild for the same position and network.
- **NNUE-02** — NNUE concat order is `[stm, opp]`.
- **NNUE-03** — A king-bucket transition invalidates the affected perspective until lazy rebuild.
- **SMP-01** — Lazy SMP helpers share the intended transposition table and do not create conflicting TT ownership.
- **SMP-02** — Only the main-thread path advances the shared TT generation according to the current design.
- **WEB-01** — The standalone browser bundle initializes without an external NNUE download.
- **WEB-02** — Browser analysis and game modes continue beyond opening-book positions.
- **REG-01** — Strength-affecting changes require statistical regression evidence before promotion.
- **DOC-01** — Documented repository commands and paths must correspond to real project surfaces.
- **REL-01** — A release candidate passes the deterministic release profile and records a release manifest.

