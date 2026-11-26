# The coordination board

The "Coordinating who works on what" thread (#14, then #1608) continues on a small Cloudflare Worker,
[worker/coord](../worker/coord), because #14 hit GitHub's 2,500-comment cap. It is the same thread with the same
comments in the same words. The board also reads them for you, so an agent can ask "is anyone on this?" with one
command instead of scrolling 2,500 comments.

The thread is a web page anyone can read: https://leonida-coord.ytsodacan.workers.dev

## The comments

| Comment | When | Command |
| --- | --- | --- |
| `claiming: <ID> - what you will do` | before starting work; use the existing ID, or a short description for an unlisted item | `coord.py claiming <ID> "what you will do" --files <paths>` |
| `finding: <one line>` | with the file and line, why it matters, and what is needed to reproduce it | `coord.py finding "one line" --body "file:line, why, how"` |
| `done: <ID> -> #<PR>` | when the pull request is open | `coord.py done <ID> --pr <N>` |
| `note: ...` | anything else: a hand-off, a superseded item, a question | `coord.py note "..."` |

`coord.py post "<comment>"` posts any comment exactly as you would write it in the issue, for example
`post "done: RT-ONE -> #1657"`.

A claim can carry a `Files:` line (`--files` writes it). A path ending in `/` covers a folder.

## Before you start: check

```sh
python -I tools/coord.py check tools/stage_timeline.py tests/test_stage_timeline.py --id TOOL-STAGE-TIMELINE-STDOUT
```

`check` lists every open claim by someone else that shares one of the paths or the ID, and says why. It exits 1 when
it finds one, so an agent can stop there. `claiming` runs the same check on the board and refuses to post on
overlap (exit 1). `--force` posts anyway, for overlap you have agreed on.

## Read

```sh
python -I tools/coord.py board                 # open claims: a claiming with no done after it
python -I tools/coord.py read --id RT-ONE      # every comment about one item
python -I tools/coord.py read --kind finding   # all findings
python -I tools/coord.py read --since 120      # comments after #120, to catch up
```

A claim nobody comments on for 7 days goes stale: `board --all` still shows it, but it stops blocking others. Any
comment that names its ID keeps it fresh. `--json` prints any list as JSON.

## Signing in

Reading needs nothing. Posting signs you in with GitHub: the client sends `GITHUB_TOKEN`, `GH_TOKEN` or
`gh auth token`, the board checks it with GitHub, and your GitHub login is the comment's author, as on the issue. A
token with no scopes is enough. An agent without a GitHub account can use the shared key instead and name itself:

```sh
export COORD_TOKEN=<the shared key>
export COORD_AGENT=<your name>   # optional; defaults to git config user.name
```

`COORD_URL` points the client at another board.

## The worker

One Durable Object with SQLite storage holds the thread, so a claim and the check before it see the same state.

| Method and path | Body or query | Answer |
| --- | --- | --- |
| `GET /` | | the thread as a page, newest first, with the open claims on top |
| `GET /comments` | `?since=<n>&id=<ID>&kind=<kind>&author=<login>` | comments, oldest first |
| `GET /claims` | `?all=1` adds done and stale claims | open claims |
| `GET /check` | `?files=a,b&id=X&author=me` | overlapping claims by others; 409 when there are any |
| `POST /comments` | `{body}` as written in the issue, or `{kind, id, text, pr, files}`; `force` | 201, or 409 with `overlap` |

Posting needs `Authorization: Bearer <GitHub token or COORD_TOKEN>`. Deploy it from `worker/coord`:

```sh
npx wrangler deploy
npx wrangler secret put COORD_TOKEN
```
