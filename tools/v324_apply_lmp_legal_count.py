from pathlib import Path

p = Path("engine/c/zchezz_v323/search.c")
s = p.read_text(encoding="utf-8")

old = """                if (!in_check && is_quiet && depth<=7 && legal_count>0 && !is_killer) {
                    quiet_count++;
                    int lmp_lim = lmp_limit[depth<8?depth:7];
"""

new = """                if (!in_check && is_quiet && depth<=7 && legal_count>0 && !is_killer) {
                    /* LMP budget must count legal quiets only. Move generation is
                     * pseudo-legal, so an illegal quiet must not consume a slot. */
                    board_make(b, m);
                    int lmp_mover_col = b->turn ^ 24;
                    int lmp_king_sq = lmp_mover_col == COL_W ? b->wk : b->bk;
                    int lmp_legal = !board_is_attacked(b, lmp_king_sq, b->turn);
                    board_unmake(b);
                    if (!lmp_legal) continue;

                    quiet_count++;
                    int lmp_lim = lmp_limit[depth<8?depth:7];
"""

if s.count(old) != 1:
    raise SystemExit(f"LMP anchor count={s.count(old)}, expected 1")

p.write_text(s.replace(old, new, 1), encoding="utf-8")
print("v3.24 LMP legal-count patch applied")
