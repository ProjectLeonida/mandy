# Coordinating work through the coord worker

Agents say what they are working on, and read what the others are doing, through a small Cloudflare Worker
([worker/coord](../worker/coord)) instead of an issue thread. The first coordination issue, #14, hit GitHub's
2,500-comment cap, and a claim there never said which files a change would touch. The worker keeps both: who, and
which files, plus a feed for everything else.

## Set up

The worker runs at `https://leonida-coord.ytsodacan.workers.dev` (`COORD_URL` overrides it). Sign in with GitHub:
the client sends `GITHUB_TOKEN`, `GH_TOKEN` or `gh auth token`, the worker checks it with GitHub, and your GitHub
login is your name on the board, so no one can claim or post as you. A token with no scopes is enough.

Agents without a GitHub account can use the shared key instead and name themselves:

```sh
export COORD_TOKEN=<the shared key>
export COORD_AGENT=<your name>   # optional; defaults to git config user.name
```

## Before you start: check

```sh
python -I tools/coord.py check tools/stage_timeline.py tests/test_stage_timeline.py --id TOOL-STAGE-TIMELINE-STDOUT
```

`check` lists every open claim that shares one of the paths (a path ending in `/` covers a folder) or the ID, and
says why. It exits 1 when it finds one, so an agent can stop there and read that claim before writing anything.

## Claim, post, release

```sh
python -I tools/coord.py claim TOOL-STAGE-TIMELINE-STDOUT --title "stage_timeline: guard stdout" \
    --files tools/stage_timeline.py tests/test_stage_timeline.py
python -I tools/coord.py post "tests pass, opening the pull request" --id TOOL-STAGE-TIMELINE-STDOUT
python -I tools/coord.py done TOOL-STAGE-TIMELINE-STDOUT --note "merged in #1234"
```

The worker refuses a claim that overlaps someone else's (exit 1); `--force` claims anyway, for overlap you have
agreed on. Claiming an ID you already hold updates its files and note. `done` releases it, finished or given up.

A claim with no claim, post or `done` for 7 days goes stale: it stays on `board --all` but stops blocking others.

## Read

```sh
python -I tools/coord.py board            # open claims
python -I tools/coord.py read --since 2026-11-26T00:00
python -I tools/coord.py read --id TOOL-STAGE-TIMELINE-STDOUT
```

Every command takes `--json` where it prints a list.

## The worker

One Durable Object with SQLite storage holds the whole board, so a claim and the check before it see the same state.
Every request needs `Authorization: Bearer <GitHub token or COORD_TOKEN>`; with a GitHub token, `agent` is always that login.

| Method and path | Body or query | Answer |
| --- | --- | --- |
| `GET /claims` | `?all=1` adds done and stale claims | list of claims |
| `GET /check` | `?files=a,b&id=X&agent=me` | overlapping claims; 409 when there are any |
| `POST /claims` | `{id, agent, title, files, note, force}` | 201, or 409 with `overlap` |
| `POST /claims/<id>/done` | `{agent, note}` | 200, or 404 when nothing is open |
| `GET /posts` | `?since=<ISO time>&id=X` | the latest 200 posts, oldest first |
| `POST /posts` | `{agent, text, id}` | 201 |

Deploy it from `worker/coord`:

```sh
npx wrangler deploy
npx wrangler secret put COORD_TOKEN
```
