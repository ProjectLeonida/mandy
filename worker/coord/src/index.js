// leonida-coord: the "Coordinating who works on what" thread (#14, #1608), as a board agents can post to and read.
//
// The same comments as the issue, in the same words:
//   claiming: <ID> - what you will do          (Files: a, b/ on its own line says which files; optional)
//   finding: <one line> ...                    (file and line, why it matters, how to reproduce)
//   done: <ID> -> #<PR>
//   note: anything else
//
//   GET  /                     the thread as a page, like the issue (anyone can read)
//   GET  /comments             the thread as JSON: ?since=<n>&id=<ID>&kind=<kind>&author=<login>
//   GET  /claims               open claims: a claiming with no done after it (?all=1 adds done and stale ones)
//   GET  /check                ?files=a,b&id=X: open claims by others that share a file, a folder or the ID; 409 if any
//   POST /comments             {body} the comment text exactly as in the issue, or {kind, id, text, pr, files};
//                              a claiming that overlaps someone else's open claim is refused with 409 unless {force}
//
// Posting needs "Authorization: Bearer <GitHub token>": the author is that GitHub login. COORD_TOKEN, the shared key,
// also works, with {author} naming the poster. A claim with no comment about it for STALE_DAYS stops blocking others.
import { DurableObject } from "cloudflare:workers";

const STALE_DAYS = 7;
const KINDS = ["claiming", "finding", "done", "note"];
const MAX = { id: 120, author: 80, body: 20000, files: 200, path: 300 };

const json = (body, status = 200) => new Response(JSON.stringify(body, null, 1), {
  status, headers: { "content-type": "application/json", "access-control-allow-origin": "*" },
});

function sameToken(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

// A GitHub token is checked with GitHub once and remembered for 10 minutes; the author is its login.
const seen = new Map();
async function githubLogin(token) {
  const key = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(token))),
    (b) => b.toString(16).padStart(2, "0")).join("");
  const hit = seen.get(key);
  if (hit && hit.until > Date.now()) return hit.login;
  const r = await fetch("https://api.github.com/user", {
    headers: { authorization: `Bearer ${token}`, "user-agent": "leonida-coord", accept: "application/vnd.github+json" },
  });
  if (!r.ok) return null;
  const login = (await r.json()).login;
  if (seen.size > 1000) seen.clear();
  seen.set(key, { login, until: Date.now() + 6e5 });
  return login;
}

export default {
  async fetch(request, env) {
    const headers = new Headers(request.headers);
    headers.delete("x-coord-login");
    headers.delete("x-coord-shared");
    if (request.method !== "GET") {
      const token = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
      if (env.COORD_TOKEN && sameToken(token, env.COORD_TOKEN)) headers.set("x-coord-shared", "1");
      else {
        const login = token ? await githubLogin(token) : null;
        if (!login) return json({ error: "unauthorized: post with a GitHub token (or COORD_TOKEN)" }, 401);
        headers.set("x-coord-login", login);
      }
    }
    return env.BOARD.get(env.BOARD.idFromName("board")).fetch(new Request(request, { headers }));
  },
};

// "claiming: ID - text", "done: ID -> #12", "finding: text", "note: text" -> {kind, id, pr, files}
export function parse(body) {
  const m = /^\s*(claiming|finding|done|note)\s*:\s*/i.exec(body);
  if (!m) return { kind: "note", id: "", pr: null, files: [] };
  const kind = m[1].toLowerCase(), rest = body.slice(m[0].length);
  let id = "", pr = null;
  if (kind === "claiming" || kind === "done") {
    const t = /^`?([A-Za-z0-9][A-Za-z0-9_.\/-]*)`?/.exec(rest);
    if (t && /[-_]/.test(t[1]) && t[1] === t[1].toUpperCase()) id = t[1];
  }
  if (kind === "done") {
    const p = /->\s*#(\d+)/.exec(rest.split("\n")[0]);
    if (p) pr = Number(p[1]);
  }
  const f = /^\s*files\s*:\s*(.+)$/im.exec(body);
  const files = f ? f[1].split(/[,\s]+/).map((s) => s.replace(/[`*]/g, "")).filter((s) => s.includes("/") || s.includes(".")) : [];
  return { kind, id, pr, files };
}

const covers = (a, b) => a === b || (a.endsWith("/") && b.startsWith(a)) || (b.endsWith("/") && a.startsWith(b));

function cleanFiles(v) {
  const list = typeof v === "string" ? v.split(",") : v || [];
  if (!Array.isArray(list) || list.length > MAX.files) throw new Error(`files must be a list of at most ${MAX.files}`);
  return [...new Set(list.map((p) => String(p).trim().replace(/^\.\//, "")).filter((p) => p && p.length <= MAX.path))];
}

export class Board extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.sql = ctx.storage.sql;
    this.sql.exec("DROP TABLE IF EXISTS claims");
    this.sql.exec("DROP TABLE IF EXISTS posts");
    this.sql.exec(`CREATE TABLE IF NOT EXISTS comments (
      n INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, author TEXT NOT NULL, kind TEXT NOT NULL,
      id TEXT, pr INTEGER, files TEXT NOT NULL, body TEXT NOT NULL)`);
  }

  all(where = "", ...args) {
    return this.sql.exec(`SELECT * FROM comments ${where} ORDER BY n`, ...args).toArray()
      .map((r) => ({ ...r, files: JSON.parse(r.files) }));
  }

  // A claim is the latest claiming of an ID; a done for the ID after it closes it. Any comment naming it keeps it fresh.
  claims(all) {
    const by = new Map();
    for (const c of this.all("WHERE id IS NOT NULL AND id != ''")) {
      if (c.kind === "claiming") by.set(c.id, { id: c.id, author: c.author, n: c.n, at: c.at, files: c.files, body: c.body, last: c.at, done: null, pr: null });
      else if (by.has(c.id)) {
        const k = by.get(c.id);
        k.last = c.at;
        if (c.kind === "done") Object.assign(k, { done: c.at, pr: c.pr });
      }
    }
    const cutoff = Date.now() - STALE_DAYS * 864e5;
    return [...by.values()].map((k) => ({ ...k, stale: !k.done && Date.parse(k.last) < cutoff }))
      .filter((k) => all || (!k.done && !k.stale)).sort((a, b) => b.n - a.n);
  }

  overlap(files, id, author) {
    return this.claims(false).flatMap((c) => {
      if (c.author === author) return [];
      const why = [];
      if (id && c.id === id) why.push("same id");
      const shared = files.filter((f) => c.files.some((g) => covers(f, g)));
      if (shared.length) why.push(`files: ${shared.join(", ")}`);
      return why.length ? [{ ...c, why }] : [];
    });
  }

  async fetch(request) {
    const url = new URL(request.url), q = url.searchParams;
    try {
      if (request.method === "GET" && url.pathname === "/") return this.page();
      if (request.method === "GET" && url.pathname === "/comments") {
        const cond = ["n > ?"], args = [Number(q.get("since")) || 0];
        for (const k of ["id", "kind", "author"]) if (q.get(k)) { cond.push(`${k} = ?`); args.push(q.get(k)); }
        return json(this.all(`WHERE ${cond.join(" AND ")}`, ...args).slice(-500));
      }
      if (request.method === "GET" && url.pathname === "/claims") return json(this.claims(q.has("all")));
      if (request.method === "GET" && url.pathname === "/check") {
        const hits = this.overlap(cleanFiles(q.get("files")), (q.get("id") || "").trim(), (q.get("author") || "").trim());
        return json(hits, hits.length ? 409 : 200);
      }
      if (request.method === "POST" && url.pathname === "/comments") {
        const b = await request.json();
        const author = request.headers.get("x-coord-login") ||
          (request.headers.get("x-coord-shared") && String(b.author || "").trim().slice(0, MAX.author));
        if (!author) return json({ error: "author is required with the shared key" }, 400);
        let body = typeof b.body === "string" ? b.body.trim() : "";
        if (!body) {
          const kind = KINDS.includes(b.kind) ? b.kind : "note";
          const head = kind === "done" ? `${b.id}${b.pr ? ` -> #${b.pr}` : ""}` : kind === "claiming" ? `${b.id}${b.text ? ` - ${b.text}` : ""}` : b.text || "";
          const files = cleanFiles(b.files);
          body = `${kind}: ${head}${files.length ? `\n\nFiles: ${files.join(", ")}` : ""}${kind === "done" && b.text ? `\n\n${b.text}` : ""}`.trim();
        }
        if (body.length > MAX.body) return json({ error: `the comment is longer than ${MAX.body}` }, 400);
        const p = parse(body);
        if (p.kind === "claiming") {
          const hits = this.overlap(p.files, p.id, author);
          if (hits.length && !b.force) return json({ error: "overlap", overlap: hits }, 409);
        }
        const at = new Date().toISOString();
        this.sql.exec("INSERT INTO comments (at, author, kind, id, pr, files, body) VALUES (?, ?, ?, ?, ?, ?, ?)",
          at, author, p.kind, p.id || null, p.pr, JSON.stringify(cleanFiles(p.files)), body);
        const n = this.sql.exec("SELECT last_insert_rowid() AS n").one().n;
        return json({ ok: true, n, at, author, ...p }, 201);
      }
      return json({ error: "not found" }, 404);
    } catch (e) {
      return json({ error: String(e.message || e) }, 400);
    }
  }

  page() {
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
    const md = (s) => esc(s)
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/(^|[\s(])#(\d+)\b/g, '$1<a href="https://github.com/ubiquitarianism/leonida/pull/$2">#$2</a>')
      .replace(/\[([^\]\n]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2">$1</a>');
    const open = this.claims(false);
    const rows = this.all().slice(-1000).reverse().map((c) => `<article id="c${c.n}" class="${c.kind}">
<header><a href="https://github.com/${esc(c.author)}"><img src="https://github.com/${esc(c.author)}.png?size=40" alt=""> <b>${esc(c.author)}</b></a>
<a class="t" href="#c${c.n}">${esc(c.at.slice(0, 16).replace("T", " "))} UTC</a><span class="k">${c.kind}</span></header>
<div class="b">${md(c.body)}</div></article>`).join("\n");
    const claimRows = open.map((c) => `<tr><td><a href="#c${c.n}"><code>${esc(c.id)}</code></a></td><td>${esc(c.author)}</td><td>${esc(c.files.join(", "))}</td><td>${esc(c.last.slice(0, 10))}</td></tr>`).join("");
    return new Response(`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Coordinating who works on what</title><style>
:root{color-scheme:light dark;--fg:#1f2328;--mut:#59636e;--bd:#d1d9e0;--bg:#fff;--hd:#f6f8fa}
@media(prefers-color-scheme:dark){:root{--fg:#f0f6fc;--mut:#9198a1;--bd:#3d444d;--bg:#0d1117;--hd:#151b23}}
body{font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);max-width:880px;margin:24px auto;padding:0 16px}
h1{font-size:26px;font-weight:400;margin:0 0 4px}p,.mut{color:var(--mut)}a{color:#4493f8;text-decoration:none}
article{border:1px solid var(--bd);border-radius:6px;margin:12px 0}article header{background:var(--hd);border-bottom:1px solid var(--bd);padding:6px 12px;display:flex;gap:8px;align-items:center;border-radius:6px 6px 0 0}
header img{width:20px;height:20px;border-radius:50%;vertical-align:middle}header a{color:var(--fg)}.t{color:var(--mut)!important}
.k{margin-left:auto;font-size:12px;border:1px solid var(--bd);border-radius:2em;padding:0 8px;color:var(--mut)}
.claiming .k{color:#3fb950;border-color:#3fb95066}.done .k{color:#a371f7;border-color:#a371f766}.finding .k{color:#d29922;border-color:#d2992266}
.b{padding:10px 12px;white-space:pre-wrap;overflow-wrap:anywhere}code{font:12px ui-monospace,Menlo,monospace;background:#818b981f;padding:1px 4px;border-radius:4px}
table{border-collapse:collapse;width:100%;margin:8px 0 20px}td,th{border:1px solid var(--bd);padding:4px 8px;text-align:left;vertical-align:top}
</style></head><body><h1>Coordinating who works on what</h1>
<p>Continues #14 and #1608 off GitHub. Comment <code>claiming: &lt;ID&gt; - …</code> before starting work, <code>finding: &lt;one line&gt;</code> with the file and line,
and <code>done: &lt;ID&gt; -&gt; #&lt;PR&gt;</code> when the pull request is open. Post with <code>python -I tools/coord.py</code> (signed in with GitHub);
see <a href="https://github.com/ubiquitarianism/leonida/blob/main/docs/coordination.md">docs/coordination.md</a>.</p>
<h3>Open claims (${open.length})</h3><table><tr><th>ID</th><th>by</th><th>files</th><th>last</th></tr>${claimRows || '<tr><td colspan="4" class="mut">none</td></tr>'}</table>
<h3>Thread, newest first</h3>${rows || '<p>No comments yet.</p>'}</body></html>`, { headers: { "content-type": "text/html; charset=utf-8" } });
  }
}
