PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then echo python3; else echo python; fi)

.PHONY: serve audit test

# Start the mock client API on http://127.0.0.1:8099
serve:
	$(PYTHON) -m uvicorn mock_api.server:app --host 127.0.0.1 --port 8099 --reload

# Peek at the side effects your skill has produced against a running server
audit:
	@curl -s http://127.0.0.1:8099/_audit | $(PYTHON) -m json.tool

# Run the offline test suite (mocked LLM + HTTP — no API key required)
test:
	$(PYTHON) -m pytest -v
