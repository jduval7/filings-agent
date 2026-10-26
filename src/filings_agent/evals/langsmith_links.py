"""Deep links into the LangSmith workspace: the pages the demo points at, looked up by name.

Read-only: nothing here creates or changes anything in LangSmith. Experiments are found by name
prefix (the latest of each), so links stay right after a sweep is re-run. Used by
`make demo-links` (printed) and by the demo server (one "In LangSmith" strip per scene).
"""

# LangSmith's client is only partly typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any

from langsmith.utils import LangSmithNotFoundError

BASE = "https://smith.langchain.com"

# Public links (no LangSmith account needed), created 2026-10-22. Shared on purpose: SEC filings
# are public and the traces contain no secrets.
PUBLIC = {
    "trace_agent_wrong": "https://smith.langchain.com/public/1c207dce-2875-48fb-a100-187d7a8b25e7/r",
    "trace_research": "https://smith.langchain.com/public/54eea7f2-e96a-4131-97c4-5ed9403b3853/r",
    "trace_fanout": "https://smith.langchain.com/public/972f1da8-670d-4791-a8b7-ce171136a3d4/r",
    "trace_publish": "https://smith.langchain.com/public/7544fb0f-87bc-4c77-a9f8-06894c7a78ef/r",
    "dataset_reconcile": "https://smith.langchain.com/public/b8630805-d17a-4ff7-bc68-f79c277f558f/d",
    "dataset_extract_test": "https://smith.langchain.com/public/f0c2c86f-329e-4e34-90e1-5ac319056b42/d",
    "dataset_research_hard": "https://smith.langchain.com/public/5b1f07b7-926a-4dbd-b2ce-5885b70c159f/d",
    "dataset_research_fresh": "https://smith.langchain.com/public/97aad40a-0861-4a0a-8420-2c125ee7688b/d",
}
# The same recorded traces inside the workspace (run ids in project filings-agent).
TRACES = {
    "trace_agent_wrong": "01a160a0-15f7-7042-a544-942a824ba534",
    "trace_research": "01a16142-aeb1-7ad1-bf7c-45c4c605ecae",
    "trace_fanout": "7e8e6fe1-6f70-42f8-99c9-76cfbdc5c66d",
    "trace_publish": "90a24716-aed0-4c03-aaee-2bfe7d0f6d70",
}
# name -> (dataset, experiment name prefixes, in the order they're compared)
COMPARISONS: dict[str, tuple[str, list[str]]] = {
    "compare_pipeline_agent": (
        "filings-agent/reconcile-dev",
        ["reconcile-claude-sonnet-4.6-v2", "reconcile-agent-"],
    ),
    "compare_ranking_flips": (
        "filings-agent/reconcile-dev",
        ["reconcile-claude-sonnet-4.6-v2", "reconcile-gpt-oss-120b-v2", "reconcile-nova-lite-v2"],
    ),
    "compare_heldout": (
        "filings-agent/extract-test",
        ["extract-gpt-oss-120b-v2", "extract-claude-haiku-4.5-v2"],
    ),
    "compare_research_hard": ("filings-agent/research-hard", ["research-"]),
    "compare_research_fresh": ("filings-agent/research-fresh", ["research-"]),
    "compare_trajectory": ("filings-agent/e2e-dev", ["e2e"]),
    "compare_memo_judged": (
        "filings-agent/memo-dev",
        ["memo-claude-sonnet-4.6", "memo-claude-haiku-4.5"],
    ),
}


def _latest(client: Any, dataset_id: Any, prefix: str) -> Any | None:
    runs = [
        p
        for p in client.list_projects(reference_dataset_id=dataset_id)
        if p.name.startswith(prefix)
    ]
    return max(runs, key=lambda p: p.start_time) if runs else None


def project_url(client: Any, project: str) -> tuple[str, str] | None:
    """(org id, project URL), or None if the project doesn't exist yet."""
    try:
        proj = client.read_project(project_name=project)
    except Exception:  # not created until its first trace
        return None
    return str(proj.tenant_id), f"{BASE}/o/{proj.tenant_id}/projects/p/{proj.id}"


def queue_url(client: Any, org: str, name: str) -> str | None:
    for q in client.list_annotation_queues(name=name):
        return f"{BASE}/o/{org}/annotation-queues/{q.id}"
    return None


def workspace_links(client: Any, project: str = "filings-agent") -> dict[str, str]:
    """Every workspace page the demo uses that exists, by key. Missing ones are left out."""
    out: dict[str, str] = {}
    found = project_url(client, project)
    if found is None:
        return out
    org, url = found
    out["project"] = url
    for key, run_id in TRACES.items():
        out[key] = f"{url}/r/{run_id}"
    for key, (dataset, prefixes) in COMPARISONS.items():
        try:
            ds = client.read_dataset(dataset_name=dataset)
        except LangSmithNotFoundError:  # not uploaded in this workspace: no link
            continue
        exps = [e for p in prefixes if (e := _latest(client, ds.id, p))]
        if exps:
            ids = ",".join(str(e.id) for e in exps)
            out[key] = f"{BASE}/o/{org}/datasets/{ds.id}/compare?selectedSessions={ids}"
    try:
        corr = client.read_dataset(dataset_name="filings-agent/reconcile-corrections")
        out["corrections"] = f"{BASE}/o/{org}/datasets/{corr.id}"
    except LangSmithNotFoundError:  # no corrections exported yet
        pass
    if q := queue_url(client, org, "filings-agent: review"):
        out["queue"] = q
    return out
