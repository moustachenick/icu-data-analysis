# Lessons

## 2026-10-18 — Verify documented behavior in the code before relying on it

- Mistake: I said classification mode "may still ask questions with `input()`", based on a
  CLAUDE.md note. The code had no `input()` calls since commit 41cbd28 (config.toml refactor).
- Rule: before I state how the code behaves, grep or read the code. Treat CLAUDE.md, summary.md
  and code comments as hints that can be stale (the same session also found a stale test and a
  code comment that pointed to a missing notes file).
