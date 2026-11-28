# Public GitHub metadata exporter

A standalone, read-only proof of concept for [issue #563](https://github.com/ubiquitarianism/leonida/issues/563).
It queries GitHub's **public REST API** and writes JSON containing public issues, pull requests,
issue/PR conversation comments, PR reviews, original IDs/URLs, authors, and timestamps.
It does **not** clone repositories, download game data, or access private content.

## Usage

Python 3.9+; standard library only:

```sh
python scripts/export_github_metadata.py ubiquitarianism/leonida --output metadata.json
```

Requests are unauthenticated, including when `GITHUB_TOKEN` is set, so private repositories
cannot be exported. GitHub's lower unauthenticated rate limit applies.

Offline tests (mocked API; no network calls):

```sh
python -m unittest discover -s tests -p 'test_export_github_metadata.py'
```

The exporter walks public issue records (including PRs), their issue comments, and PR reviews,
following numbered pages of up to 100 records. It prints summary counts after writing.
On API errors or duplicate record IDs, it exits unsuccessfully without publishing a partial
new output file. A pre-existing output file remains unchanged on failure.

## Limitations

* This is **not** a Radicle importer or a full GitHub archive. It excludes review-thread
  comments, commit discussions, attachments, labels, milestones, events, deleted records,
  private issues, and content inaccessible through the public API.
* The export is not a transactional snapshot: discussions may change while pagination runs.
  Counts validate what was retrieved, **not** completeness against GitHub's live totals.
* GitHub secondary rate limits may require rerunning the export; the retry budget is bounded.
* Only export information you are authorized to access. Review retention and privacy needs
  before publishing the JSON output.
