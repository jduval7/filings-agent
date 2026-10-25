"""Print every LangSmith page the demo visits, as clickable links.

    make demo-links

Looks things up by name (latest experiment for each prefix), so the links stay right after
re-running a sweep. Read-only: it never creates or changes anything in LangSmith. The same links
appear in the review screen's Demo panel when `make demo` is connected to LangSmith.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langsmith import Client

from filings_agent.evals.langsmith_links import PUBLIC, workspace_links


def main() -> None:
    load_dotenv()
    links = workspace_links(Client(), os.environ.get("LANGSMITH_PROJECT", "filings-agent"))
    print("\n=== Your workspace ===")
    for key, url in links.items():
        print(f"  {key}:\n    {url}")
    print("\n=== Public (no LangSmith account needed) ===")
    for key, url in PUBLIC.items():
        print(f"  {key}:\n    {url}")


if __name__ == "__main__":
    main()
