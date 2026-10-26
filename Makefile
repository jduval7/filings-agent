.PHONY: setup test check cov data serve gate studio demo-links web web-dev web-check smoke receipt fresh-check audit web-e2e demo demo-check demo-record demo-studio import-qa

setup:        ## install deps (Python and the review UI)
	uv sync
	cd web && npm ci

test:         ## unit tests (no network, no LLM)
	uv run pytest

check:        ## lint + strict types + tests
	uv run ruff check . && uv run pyright && uv run pytest

cov:          ## tests with coverage; fails under 95%
	uv run pytest --cov=filings_agent --cov-report=term-missing --cov-fail-under=95

data:         ## fetch SEC filings, build answer key + eval datasets (needs SEC_USER_AGENT)
	@uv run python -m filings_agent.preflight "make data" sec
	uv run filings-agent build-data

demo:         ## the walkthrough at http://localhost:8000: no AWS, LangSmith or network needed ($0)
	@test -f web/dist/index.html || $(MAKE) web
	uv run python scripts/demo.py serve --reset

demo-check:   ## every demo scene seeded offline and checked (what `make demo` will show)
	uv run python scripts/demo.py check

demo-record:  ## author only: re-record demo/data from data/ (needs AWS_PROFILE; cents)
	@uv run python -m filings_agent.preflight "make demo-record" data aws
	uv run python scripts/demo.py record

serve:        ## HTTP API on :8000 (needs AWS_PROFILE for Bedrock)
	@uv run python -m filings_agent.preflight "make serve" aws
	uv run filings-agent serve

gate:         ## eval regression gate (cache-backed; fails if quality drops below configs/eval_gate.yaml)
	@uv run python -m filings_agent.preflight "make gate" data cache
	uv run filings-agent gate

studio:       ## LangGraph Studio server (open https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)
	@uv run python -m filings_agent.preflight "make studio" aws
	LANGSMITH_TRACING=true LANGSMITH_PROJECT=filings-agent uv run langgraph dev --no-browser --port 2024

demo-studio:  ## LangGraph Studio on the demo recordings (no Bedrock; graph "filings_agent")
	@# LangSmith settings from .env if there is one (traces); the demo still never calls Bedrock
	set -a; [ -f .env ] && . ./.env; set +a; uv run langgraph dev --config langgraph.demo.json --no-browser --port 2024

import-qa:    ## QA's catches in LangSmith (dataset filings-agent/reconcile-qa) -> committed regression tests
	@uv run python -m filings_agent.preflight "make import-qa" langsmith
	uv run filings-agent import-qa

demo-links:   ## every LangSmith page the demo visits, in order (read-only)
	uv run python scripts/demo_links.py

WEB_DEPS = web/node_modules/.package-lock.json
$(WEB_DEPS): web/package-lock.json
	cd web && npm ci

web: $(WEB_DEPS)  ## build the review UI (served by `make serve` at http://localhost:8000)
	cd web && npm run build

web-dev: $(WEB_DEPS)  ## review UI with hot reload on :5173 (run `make serve` alongside)
	cd web && npm run dev

web-check: $(WEB_DEPS)  ## review UI: strict types + tests
	cd web && npm run typecheck && npm test

smoke:        ## end to end over HTTP: real server, graph and (cached) models; ~$0 (needs AWS_PROFILE)
	@uv run python -m filings_agent.preflight "make smoke" data cache aws
	uv run python scripts/smoke.py --langsmith

receipt:      ## the docs' headline numbers, recomputed from the cache ($0; paid calls blocked)
	@uv run python -m filings_agent.preflight "make receipt" data cache
	uv run python scripts/receipt.py --recompute

fresh-check:  ## CI on a fresh clone: copy without gitignored files (no data/, .env, .venv), run every check
	@rm -rf /tmp/filings-agent-fresh && mkdir -p /tmp/filings-agent-fresh
	(git ls-files; git ls-files --others --exclude-standard) | rsync -a --files-from=- ./ /tmp/filings-agent-fresh/
	cd /tmp/filings-agent-fresh && env -u SEC_USER_AGENT -u LANGSMITH_API_KEY -u AWS_PROFILE -u LANGSMITH_TRACING \
	  AWS_CONFIG_FILE=/dev/null AWS_SHARED_CREDENTIALS_FILE=/dev/null AWS_EC2_METADATA_DISABLED=true \
	  sh -c 'uv sync --quiet && make check && make cov && make demo-check && cd web && npm ci --silent && npm run typecheck && npm test && npm run build && cd .. && \
	  for t in data receipt gate smoke import-qa demo-record serve studio; do \
	    make $$t > /tmp/fresh-$$t.log 2>&1; rc=$$?; \
	    if [ $$rc -ne 2 ] || grep -q Traceback /tmp/fresh-$$t.log; then echo "make $$t: no clear message (exit $$rc)"; exit 1; fi; \
	  done && echo "fresh clone: every target works or explains what it needs"'

audit:        ## known-vulnerability scan (Python + npm) and a security lint of our code
	uv export --frozen --no-hashes --all-groups --no-emit-project > /tmp/filings-agent-reqs.txt
	uvx pip-audit -r /tmp/filings-agent-reqs.txt --no-deps --disable-pip
	cd web && npm audit --audit-level=low
	uvx bandit -r src -q

web-e2e: $(WEB_DEPS)  ## the review UI in a real browser against the real server (needs AWS_PROFILE; ~$0 cached)
	@uv run python -m filings_agent.preflight "make web-e2e" data cache aws
	cd web && npm run build && npx playwright test

# Local-only targets (the submission video's tooling; not part of the project, gitignored).
-include video.mk
