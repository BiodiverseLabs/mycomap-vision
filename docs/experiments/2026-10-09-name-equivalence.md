---
title: 'Name equivalence: strict, sensu lato and species complexes'
slug: name-equivalence
date: '2026-10-09'
status: adopted
question: How should a near miss be scored when the name is a matter of taxonomy (genus splits, gender
  endings, species complexes) rather than a wrong identification?
branch: feat/name-equivalence (merged c7b70e0)
commits:
- 249db45
- fd61ed1
- c7b70e0
benchmark: 99-record held-out audit; heldout-2026-10-08 dev and test
split: dev; test
model: bioclip-2-ft-20261007-165400
methods:
- nearest
- nearest+mean
headline: 'Held-out test: species s.l. equals strict to within 0.1 point; species complex adds about 3
  points at top 1; genus s.l. adds 0.6-0.7.'
verdict: Gender-ending folding and genus groups move almost nothing for Vision; complex scoring flatters
  outside models without provisional names.
decision: 'Steve 2026-10-08/09: adopted for scoring only (never training labels), beside strict and never
  instead of it; temporary codes score like names; gender endings fold in s.l. and complex.'
related:
- heldout-benchmark
- published-danish-models
---

## Question
What counts as the same species when comparing an answer with the DNA name?

## Why it matters
*Cortinarius* and *Inocybe* were split into several genera recently; epithets change gender
endings; provisional codes sit inside look-alike complexes. A strict match can call a sound
answer wrong.

## Setup
`name_equiv.py`: strict (Vision's own label rule); s.l. counts described epithets across the
genera of Cortinarius s.l. (9 genera) and Inocybe s.l. (5), with gender endings folded (-us/-a/-um,
-er/-ra/-rum; -is/-e apart); complex (beta) = the same epithet stem (at least 4 letters), codes
matched case-sensitively. Groups are for scoring only.

## What we tried
Joining provisional codes across genera was rejected: codes are numbered within a genus, so it
inflated matches (104 vs 49).

## Results
99-record audit: species strict / s.l. / complex 40.4 / 40.4 / 43.4; genus 77 / 80.

Held-out test (nearest): species strict 48.3, s.l. 48.3, complex 51.3 at top 1; genus 80.1 vs s.l.
80.8. With the blend: 54.2 / 54.2 / 57.2; genus 83.5 / 84.1.

For an outside model without provisional codes, complex scoring credits European parent names
against North American codes split from them: DF20 on all development records with a
species-level truth (n 2,918) 7.8% strict vs 12.7% complex; on formal names only 12.6 vs 13.3.

## Verdict
Report all three beside strict; for outside models, report complex on formal names only.

## Decision
Adopted for scoring (Steve, 2026-10-08/09).

## Next
Keep the genus groups in step with the taxonomy; review whether complex should leave beta.
