# General rules

How agents work in the Resonance repo. They apply to every task, alongside `CLAUDE.md` and the skills in `.claude/skills/`.

## Scope

- Do the requested task, nothing adjacent. Leave the code, configs, lockfiles, and formatting that the task doesn't need as
  they are.
- Ask a short, concrete question before acting on a material ambiguity: two plausible readings, a missing requirement,
  an unclear location, or a new file, folder, or move. Infer only details that can't change the result.
- A user's explicit request overrides these rules for that task.

## Code

- Between equivalent options, take the one with fewer lines and fewer abstractions, and reuse the patterns and helpers
  already in the repo.
- Put a change in an existing file when one reasonably holds it.
- Write a comment only for non-obvious reasoning, a constraint, or an invariant. Add documentation only when the task
  asks for it or a public contract changes.

## Verification

- Prove the changed behavior with the smallest check that can fail: an existing test or a focused command. Add a
  permanent test only when the change is more than a few lines and nothing else covers it.

## Before you finish

Read your diff and check:

- every changed line is needed for the task;
- nothing structural (a new file, folder, dependency, or move) went in without the user's agreement;
- the change was verified;
- anything critical or unexpected you found is recorded in `knowledge/`.

## Outdated instructions

Skills, `CLAUDE.md`, and these rules describe the repo as it was when they were written. When one of them disagrees with
the code, a note, or what you observe (a flag that no longer exists, a path that moved, a step that no longer works),
say so to the user: name the file, what's wrong, and the fix you propose. Edit it only after the user agrees, and
then follow the corrected version.

## Answers

- Lead with the conclusion. For a yes-or-no claim, answer **yes**, **mostly yes**, **partly**, **mostly no**, **no**, or
  **unknown**.
- Back it with numbers or a concrete example. Say "unknown" when the evidence is missing.
- Mention a caveat only when it changes the conclusion. Give the main conclusion and its evidence in one coherent order,
  not a chain of increasingly important side claims.

## knowledge/

`knowledge/` is the shared memory of the people and agents on this project. It holds what was measured, decided, or
learned, and why.

- **Read first.** Before a non-trivial task, read `knowledge/INDEX.md`, descend through the folder indexes that bear on
  the task, and open the notes they list.
- **Record what matters later.** When you find something critical or unexpected, record it. Update an existing note when
  one covers the topic; otherwise create one.
- **Keep the indexes whole.** Every folder has an `INDEX.md` that links its subfolders and lists each of its files on
  one line: `` `file` `` — what it answers — status. The status is `current`, `stale`, or `superseded by <path>`,
  optionally followed by a short note in parentheses; a `stale` status always gives the reason. Add, move, or
  re-status the line in the same change as the file.
- **Mark, don't delete.** When a note goes out of date, change its status and add a dated line at its top that says what
  changed. Delete a note only when the user asks.
- **Shallow folders.** While there are only a few notes, they sit flat at the root. Once a topic reaches about five
  notes (for example corpus notes, or filtering metrics), it gets its own folder with its own `INDEX.md`. Ask before
  moving existing notes, and fix every link to a moved note in the same change.

### Note style

- Title: `# <Topic>: <what it answers> (YYYY-MM-DD)`.
- Opening lines: what was measured, on which corpus, shard, and sample (rows, hours), which hardware, and with which
  config.
- State each finding as a **bold claim** followed by its numbers. Put verdict counts, percentiles, and threshold
  trade-offs in tables. Give sample sizes, because many calibration samples are small.
- Attribute decisions and listening results to "the user", with a date: `**Decision (the user, 2026-10-02):** ...`.
- Tag unfinished work in its heading: `(proposed, not implemented)`, `(not measured yet)`.
- Cite the commit, config, and script for every number. Name scripts that live outside the repo as such.
- Close with `## ⚑ Gotchas` and `## Open points` when they exist.
