# Identify result: "Your collections" and best match per photo (2026-10-18)

Steve asked (2026-10-18) for three panels below the ID area of each result card
(answer hero, safety note, photo switcher, every-rank breakdown), in this order:

1. **Your collections**: signed-in people only. Your own DNA-verified records
   among Vision's references, ranked by how alike they look to this find.
2. **Closest DNA-verified specimens from other collectors**: today's list, minus
   your own records, so nothing shows twice.
3. **Best match per photo**: each uploaded photo's single closest DNA-verified
   photo (today only reachable by clicking a photo in the switcher).

## Panel content

| Panel | Shows | Cards |
|---|---|---|
| Your collections | up to 6 of your reference records, most alike first, no threshold; each card shows its "% alike" so a weak match reads as weak | today's `SpecimenCard`, plus a "same species as the answer" tag when it is |
| Closest from other collectors | 6, as today, with your records left out | unchanged |
| Best match per photo | one card per uploaded photo: your photo thumbnail beside its closest record | `SpecimenCard`, labelled "Photo N"; marked "yours" when the record is yours |

- "How alike" is the score list 2 already uses: for each of your photos, its best
  match among the record's photos, averaged over your photos.
- Signed out: panel 1 is absent and panel 2 keeps today's heading ("Closest
  DNA-verified specimens"). "From other collectors" appears only when panel 1
  could be filled for this person.
- Best match per photo is not de-duplicated: it answers "which reference is
  nearest to this photo", whoever collected it. With one photo it is the same as
  the first card of the combined lists, so it is hidden for a one-photo find.
- Photos follow the same rules as everywhere else (`pick_shown`, permissions).
  Your own ARR photos are shown to you only if you granted permission, the same
  as for anyone; no special case.

## The join (who collected which reference)

Revised after a read-only check on prod (2026-10-18): on green iNat records
`observations.collector` is a display name on 132,753 of 164,143 rows, not a
login, so a collector join on mycomap.org finds only 49 people / 11,803 records.
Vision's own manifest already holds the owner each source reports for every
reference record (`inat_observations.user_id` / `user_login`, read from the iNat
API, for all 161,359 iNat references; `mo_observations.owner_id` / `owner_login`
for MO). So mycomap.org says who the person is on iNat and MO, and Vision matches
its references on the source's own owner field. Matching that way, 227 people have
iNat references (99,026 records), 101 of them through a verified link, and 16
people have MO references (1,080 records), with no legal-name matching needed.

**New .org endpoint** (own .org branch, key-gated like the other Vision reads):

```
GET /api/vision/linked-accounts?user=<sub>
Authorization: Bearer <VISION_API_KEY>

200 { generatedAt, user,
      inat: [ { user_id, login } ],     // verified iNat connections
      mo:   [ { user_id, login } ],     // verified MO connections
      unverified: { inat: bool, mo: bool } }   // a login typed, not connected
```

- Only verified `platform_credentials` rows count (proved by OAuth). A login in
  `users.inaturalist_username` without a verified connection is reported only as
  `unverified` and is never matched: older values predate the rule that a login
  is set only from a verified connection, so one could name someone else.
- Refusals: 503 `not_configured`, 401 without the key, with a wrong key, or with
  a member or admin session instead of the key (a session is not a key, as on
  every Vision read); 400 for a missing or malformed `user`; 404 for no such
  active user. Every answer `Cache-Control: no-store`. Listed in the auth sweep.
- 4 verified iNat logins are each connected to two .org accounts today; both
  accounts proved they own that iNat account, so both see its records.

**Vision side**

- `/api/me` already has the person (`sub`). On an Identify request with a session,
  Vision asks .org for that `sub`'s accounts, caches them per `sub` for 10
  minutes (in memory, never on disk, never logged), and picks the reference
  records whose owner is one of them: iNat by user id (a renamed login still
  matches), MO by owner id, else login. A failed read: no panel 1, list 2 as
  today; never an error for the identification.
- Scoring: the query's vectors against just that person's reference vectors
  (`reference_vectors(cols)`): median 75 records a person, at most ~8,800; CPU,
  no global top-k.
- A link identify whose observation is itself one of your references leaves it
  out, exactly as it is left out of the main answer.
- Kept answers (link identifies answered before) are shared between people, so
  nothing personal is ever stored in them: the cache keeps the unpersonalised
  answer with a few spare specimens, and panels 1 and 2 are made per request.

## Privacy

- Only the signed-in person ever gets panel 1, for their own `sub` from our
  signed session cookie. No route takes a user id from the browser.
- The answer to `/api/identify` for person A never carries person B's records,
  and kept answers carry no one's.
- MV_SIGNIN off (local, or the site open): no person, so no panel 1.

## Empty states

- Signed out: nothing (no teaser).
- Signed in, no verified iNat or MO connection (a typed login counts as none):
  one line where panel 1 would be: "Connect your iNaturalist or Mushroom Observer
  account on mycomap.org to see your own DNA-verified records here." (links to
  the .org page where accounts are connected).
- Linked, but none of your records are in Vision's reference: nothing.
- Linked, records in the reference: the panel, always (no similarity threshold).
- A find the fungus gate says is not a fungus: no panel (there are no specimens).
- mycomap.org can't be read: no panel and the usual list; never an error.
- Panels 1 and 3 belong to "All photos combined"; picking one photo in the
  switcher shows that photo's own list, as before.

## Mobile width

Same grid as today: two cards per row under 640 px, three above. Best match per
photo: one row per photo (your thumbnail 56 px, then the card), stacked.

## API shape (Vision site, `/api/identify` and `/api/identify/observation`)

Each result gains, only when a person is signed in:

```
"yours": { "linked": true, "records": 214,          // yours in the reference
           "specimens": [ <same shape as specimens>, ... up to 6 ] }
```

and `specimens` then leaves out the person's own records. Absent `yours`: signed
out or the .org read failed. `per_photo[i].specimens[0]` is the per-photo best
match (already there); each specimen gains `"yours": true` where it is.
The public `/api/v1/` answers are unchanged.

## Decisions (Steve, 2026-10-18)

- The layout as in the real-screen mockup.
- Verified connections only. Typed-but-unconnected iNat logins (126 more people
  today) get the "Connect your account" line, because an old typed value could
  name another person.
- The MO legal-name match is not needed (MO owner logins come from MO itself).
- An iNat account verified on two .org accounts (4 today) shows its records to
  both: each proved it owns it.

## Where it lives

- mycomap.org: `GET /api/vision/linked-accounts` (branch feat/vision-linked-accounts).
- Vision: `src/mycomap_vision/yours.py` (accounts, cache, owner index, the
  personal part), `Identifier.own_specimens` and `Identifier.owners`
  (identify.py), the site routes in api.py; web `lib/yours.ts` and the panels in
  `pages/Identify.tsx`. Needs MV_ORG_BASE_URL and the server key, as the
  photographers' answers already do; without them there is no panel.

## Tests

Vision: anon sees no `yours`; person A never gets B's records; ordering; list 2
de-duplicated against panel 1 and still 6 long; empty states; kept answers carry
nothing personal; .org read failure; owner match by id survives a login rename.
.org: key gate (503/401, session is not a key), 400/404, verified connections
only, a typed login only as `unverified`, no other user's accounts. Each test made to fail once.
