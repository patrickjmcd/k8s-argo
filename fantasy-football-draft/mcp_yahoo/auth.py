"""One-time Yahoo OAuth2 bootstrap.

Run this once from a terminal to complete Yahoo's 3-legged OAuth flow:

    python -m mcp_yahoo.auth

It opens (or prints) a Yahoo consent URL. Approve, paste the verifier code back,
and the resulting access/refresh token is written to your .env file
(YAHOO_ACCESS_TOKEN_JSON). After that the MCP server refreshes the token on its
own and never needs a browser again.

Requires these in the environment or .env first:
    YAHOO_CONSUMER_KEY, YAHOO_CONSUMER_SECRET, YAHOO_LEAGUE_ID
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from yfpy.query import YahooFantasySportsQuery

_ENV_DIR = Path(os.environ.get("YAHOO_ENV_DIR", Path.cwd()))


def main() -> int:
    for required in ("YAHOO_CONSUMER_KEY", "YAHOO_CONSUMER_SECRET", "YAHOO_LEAGUE_ID"):
        if not os.environ.get(required):
            print(f"error: {required} is not set (add it to .env or export it)", file=sys.stderr)
            return 1

    print("Starting Yahoo OAuth. A consent URL will be shown; approve and paste the code.")
    query = YahooFantasySportsQuery(
        league_id=os.environ["YAHOO_LEAGUE_ID"],
        game_code=os.environ.get("YAHOO_GAME_CODE", "nfl"),
        yahoo_consumer_key=os.environ["YAHOO_CONSUMER_KEY"],
        yahoo_consumer_secret=os.environ["YAHOO_CONSUMER_SECRET"],
        env_file_location=_ENV_DIR,
        save_token_data_to_env_file=True,
        # Print the URL/prompt in the terminal instead of trying to open a browser.
        browser_callback=True,
    )
    # Force a request so the token is minted and persisted.
    info = query.get_league_info()
    name = getattr(info, "name", "your league")
    print(f"\nSuccess. Authenticated to '{name}'. Token saved to {_ENV_DIR / '.env'}.")
    print("You can now run the MCP server without re-authenticating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
