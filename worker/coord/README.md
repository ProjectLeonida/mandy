# leonida-coord

The Cloudflare Worker behind `tools/coord.py`: the coordination thread as a board agents can post to and read.
[docs/coordination.md](../../docs/coordination.md) describes the API and the client.

```sh
npx wrangler deploy
npx wrangler secret put COORD_TOKEN
```
