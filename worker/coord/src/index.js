// leonida-coord: a board where agents say what they are working on, and read what the others are doing.
//
//   GET  /claims[?all=1]            open claims (with ?all=1, done and stale ones too)
//   GET  /check?files=a,b&id=X      open claims that share a file, a folder or the ID; 409 when there is overlap
//   POST /claims  {id, agent, title, files[], note, force}   claim work; 409 on overlap unless force
//   POST /claims/:id/done  {agent, note}                     release it (finished or given up)
//   GET  /posts[?since=ISO&id=X]    the message feed, newest last
//   POST /posts   {agent, text, id}                          say something, optionally about a claim
//
// Every request needs "Authorization: Bearer <COORD_TOKEN or a GitHub token>"; with a GitHub token the agent is its login. A claim not touched for STALE_DAYS stops blocking others.
import { DurableObject } from "cloudflare:workers";

const STALE_DAYS = 7;
const MAX = { id: 80, agent: 80, title: 200, note: 2000, text: 4000, files: 200, path: 300 };

const json = (body, status = 200) =>
  new Response(JSON.stringify(body, null, 1), { status, headers: { "content-type": "application/json" } });

function sameToken(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

// Who is calling: the shared COORD_TOKEN (the agent names itself), or a GitHub token, checked with GitHub once and
// remembered for 10 minutes, in which case the agent is that GitHub login and cannot be anyone else.
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
    const token = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
    let login = "";
    if (!(env.COORD_TOKEN && sameToken(token, env.COORD_TOKEN))) {
      login = token ? await githubLogin(token) : null;
      if (!login) return json({ error: "unauthorized: send COORD_TOKEN or a GitHub token" }, 401);
    }
    const headers = new Headers(request.headers);
    headers.delete("x-coord-login");
    if (login) headers.set("x-coord-login", login);
    return env.BOARD.get(env.BOARD.idFromName("board")).fetch(new Request(request, { headers }));
  },
};

const str = (v, max, name) => {
  if (v === undefined || v === null) return "";
  if (typeof v !== "string") throw new Error(`${name} must be a string`);
  if (v.length > max) throw new Error(`${name} is longer than ${max}`);
  return v.trim();
};

function cleanFiles(v) {
  if (v === undefined || v === null) return [];
  const list = typeof v === "string" ? v.split(",") : v;
  if (!Array.isArray(list) || list.length > MAX.files) throw new Error(`files must be a list of at most ${MAX.files}`);
  return [...new Set(list.map((p) => str(p, MAX.path, "file").replace(/^\.\//, "")).filter(Boolean))];
}

// A path ending in "/" is a folder and covers everything under it.
const covers = (a, b) => a === b || (a.endsWith("/") && b.startsWith(a)) || (b.endsWith("/") && a.startsWith(b));

export class Board extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.sql = ctx.storage.sql;
    this.sql.exec(`CREATE TABLE IF NOT EXISTS claims (
      id TEXT PRIMARY KEY, agent TEXT NOT NULL, title TEXT, files TEXT NOT NULL, note TEXT,
      created TEXT NOT NULL, updated TEXT NOT NULL, done TEXT)`);
    this.sql.exec(`CREATE TABLE IF NOT EXISTS posts (
      n INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, agent TEXT NOT NULL, id TEXT, text TEXT NOT NULL)`);
  }

  rows(all) {
    const cutoff = Date.now() - STALE_DAYS * 864e5;
    return this.sql.exec("SELECT * FROM claims ORDER BY updated DESC").toArray()
      .map((r) => ({ ...r, files: JSON.parse(r.files), stale: !r.done && Date.parse(r.updated) < cutoff }))
      .filter((r) => all || (!r.done && !r.stale));
  }

  overlap(files, id, agent) {
    return this.rows(false).flatMap((c) => {
      const why = [];
      if (id && c.id === id && c.agent !== agent) why.push("same id");
      const shared = files.filter((f) => c.files.some((g) => covers(f, g)));
      if (shared.length && !(id && c.id === id)) why.push(`files: ${shared.join(", ")}`);
      return why.length ? [{ ...c, why }] : [];
    });
  }

  post(agent, id, text) {
    this.sql.exec("INSERT INTO posts (at, agent, id, text) VALUES (?, ?, ?, ?)", new Date().toISOString(), agent, id || null, text);
  }

  async fetch(request) {
    const url = new URL(request.url);
    const parts = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
    try {
      const body = request.method === "POST" ? await request.json() : {};
      const login = request.headers.get("x-coord-login");
      if (login) body.agent = login;
      if (request.method === "GET" && parts[0] === "claims" && parts.length === 1)
        return json(this.rows(url.searchParams.has("all")));

      if (request.method === "GET" && parts[0] === "check") {
        const hits = this.overlap(cleanFiles(url.searchParams.get("files")), str(url.searchParams.get("id"), MAX.id, "id"),
          login || str(url.searchParams.get("agent"), MAX.agent, "agent"));
        return json(hits, hits.length ? 409 : 200);
      }

      if (request.method === "POST" && parts[0] === "claims" && parts.length === 1) {
        const id = str(body.id, MAX.id, "id"), agent = str(body.agent, MAX.agent, "agent");
        if (!id || !agent) return json({ error: "id and agent are required" }, 400);
        const files = cleanFiles(body.files), title = str(body.title, MAX.title, "title"), note = str(body.note, MAX.note, "note");
        const hits = this.overlap(files, id, agent);
        if (hits.length && !body.force) return json({ error: "overlap", overlap: hits }, 409);
        const now = new Date().toISOString();
        this.sql.exec(`INSERT INTO claims (id, agent, title, files, note, created, updated, done) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
          ON CONFLICT(id) DO UPDATE SET agent=excluded.agent, title=excluded.title, files=excluded.files,
          note=excluded.note, updated=excluded.updated, done=NULL`, id, agent, title, JSON.stringify(files), note, now, now);
        this.post(agent, id, `claimed ${id}${title ? `: ${title}` : ""}${files.length ? ` (${files.join(", ")})` : ""}`);
        return json({ ok: true, id, agent, overlap: hits }, 201);
      }

      if (request.method === "POST" && parts[0] === "claims" && parts[2] === "done" && parts.length === 3) {
        const agent = str(body.agent, MAX.agent, "agent"), note = str(body.note, MAX.note, "note");
        const found = this.sql.exec("SELECT id FROM claims WHERE id = ? AND done IS NULL", parts[1]).toArray();
        if (!found.length) return json({ error: "no open claim with that id" }, 404);
        this.sql.exec("UPDATE claims SET done = ?, updated = ? WHERE id = ?", new Date().toISOString(), new Date().toISOString(), parts[1]);
        this.post(agent || "?", parts[1], `released ${parts[1]}${note ? `: ${note}` : ""}`);
        return json({ ok: true });
      }

      if (request.method === "GET" && parts[0] === "posts") {
        const since = url.searchParams.get("since") || "", id = url.searchParams.get("id");
        const rows = id
          ? this.sql.exec("SELECT * FROM posts WHERE at > ? AND id = ? ORDER BY n DESC LIMIT 200", since, id).toArray()
          : this.sql.exec("SELECT * FROM posts WHERE at > ? ORDER BY n DESC LIMIT 200", since).toArray();
        return json(rows.reverse());
      }

      if (request.method === "POST" && parts[0] === "posts") {
        const agent = str(body.agent, MAX.agent, "agent"), text = str(body.text, MAX.text, "text");
        if (!agent || !text) return json({ error: "agent and text are required" }, 400);
        const id = str(body.id, MAX.id, "id");
        this.post(agent, id, text);
        if (id) this.sql.exec("UPDATE claims SET updated = ? WHERE id = ? AND done IS NULL", new Date().toISOString(), id);
        return json({ ok: true }, 201);
      }
      return json({ error: "not found" }, 404);
    } catch (e) {
      return json({ error: String(e.message || e) }, 400);
    }
  }
}
