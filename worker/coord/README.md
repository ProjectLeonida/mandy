# leonida-coord

The Cloudflare Worker behind `tools/coord.py`: claims and a message feed for agents working on this repository.
[docs/coordination.md](../../docs/coordination.md) describes the API and the client.

```sh
npx wrangler deploy
npx wrangler secret put COORD_TOKEN
```
