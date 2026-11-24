# Proposal: the validation route on mycomap.org's green-records feed

Status: proposed, 24 Nov 2026. Not implemented on mycomap.org. Vision reads these fields
already (`records.validation_route`) and treats their absence as "route unknown".

## Why

The paired v1 / v2 scores (`daily_scores.py`) follow the Paper Coordinator's ruling of
24 Nov 2026 (research repo f79c7d2, `paper/PLAN.md`). Condition (a) says each ledger row
records the validation route: manual, or auto with its rules version, and whether Vision was
the agreement, the penalty lift or the second source. Condition (b) says the primary
comparison leaves out Vision-assisted validations and auto-validations that Vision gated
(`agreement_validated`, `penalty_lift`, `second_source_vision`).

Steve also asked for every green record to be answered by both models, whether or not
mycomap.org called Vision when it validated it, with the records that had no Vision call
counted on their own inside the primary group.

The feed (`GET /api/vision/green-records` and `/since`, built by
`artifacts/api-server/src/services/visionGreenRecords.ts`) carries three Vision flags today:
`validated_with_vision`, `held_by_vision` and `vision_genus_only`. It also carries
`first_green_at`. It does not say:

- whether the first green validation was manual or auto, or which auto rules version made it;
- which part Vision played (the `vision_decision_events` kinds);
- whether mycomap.org called Vision for the record at all.

Until it does, the Vision box puts every record that is not `validated_with_vision` in the
"route unknown" group. Those records are answered and scored, and counted on their own, but
they stay out of the primary comparison (fail-closed). So the primary group stays empty until
mycomap.org sends the route.

## The change (additive, four fields on each feed row)

| Field | Type | Meaning |
|---|---|---|
| `first_green_method` | `"manual"` or `"auto"` | How the record's FIRST green validation was made: the `observation_validation_methods.method` of the row for the earliest green `validation_events` event (status 1 or 4), matched by observation and platform as `first_green_at` is. Left out of the row when not known. |
| `first_green_rules_version` | string or null | That row's `rules_version_name`, else `rules_version` as text ("1.4"). Null for a manual validation. |
| `vision_roles` | array of strings | The distinct `vision_decision_events.kind` values for the record (same platform rule as the flags) with `occurred_at` at or before `first_green_at`. `[]` when there are none. Kinds only: never names, genera or probabilities, as that table already guarantees. |
| `vision_called` | boolean | Whether mycomap.org asked Vision about the record at or before `first_green_at`: a `record_vision_runs` row created by then, or any `vision_decision_events` row by then. |

Kinds that decide the group on the Vision side (`daily_scores.route_of`):

- `agreement_validated`, `penalty_lift`, `second_source_vision`: Vision gated the decision.
- `vision_shown_at_manual_validation`, `read_pick`, `read_set_aside`, `read_trashed`,
  `model_guided_change`: Vision-assisted (research 900b62b, 115ef57).
- When alone, primary (no part in the decision): `skipped_for_person`,
  `vision_available_at_manual_validation` and `model_guided_review` (both counted),
  `second_source_observer`, `observer_over_vision`, `hold`, `genus_veto`,
  `no_second_source`, `flagged_read_to_person`, `vision_no_answer`,
  `vision_checked_no_effect` (research 115ef57).
- `[]`: no part. `vision_called: false` adds it to the primary's "no Vision call" sub-count.
- Any other kind: not ruled on yet, so the record is "route unknown" (never primary) until
  the Paper Coordinator rules and a commit adds the kind (`daily_scores.KNOWN_ROLES`).
- An assisted kind, then a gated one, decides the group whatever else is present.

### Notes for the implementer

- Reuse the platform matching that `first_green_at` and `VISION_FLAG` already use, and build
  the per-page sets once per page (`MATERIALIZED` CTEs). The 2026-11-01 note in
  `visionGreenRecords.ts` explains why a correlated `EXISTS` per record made pages time out.
- These must be booleans, strings and arrays only. Never null for `vision_roles` or
  `vision_called` when `first_green_method` is sent. Leave all four out together when the
  method can't be found.
- Add them to `VISION_FIELDS`, or to a sibling list, and to the `GreenRecord` type. Add tests
  in `visionGreenRecords.test.ts`: a manual row with no Vision call, an auto row gated by
  each of the three kinds, a manual row with `vision_shown_at_manual_validation`, an event
  after `first_green_at` (not counted), and a record of an unknown platform.
- No schema change is needed: everything is read from `observation_validation_methods`,
  `validation_events`, `vision_decision_events` and `record_vision_runs`.

## On the Vision side once it ships

- No setting changes. Records scored while their route was unknown are regrouped by
  `cut_routes` (insert-only) the first night the feed gives their route. Their answers never
  change.
- The internal view (`/api/daily-scores/internal`) then fills the primary group, with its
  `no_vision_call`, `vision_call_not_known` and `held_by_vision` sub-counts.
- If mycomap.org picks other field names, change the four `records.ROUTE_*` /
  `VISION_*_FIELD` constants (one line each).
