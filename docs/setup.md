# Setup

## Environment

Python 3.12 (tested with 3.12.4). pgmpy needs torch; the CPU wheel is enough and much smaller, so install it first:

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Or use Docker; the image runs the tests by default:

```bash
docker build -t aced-bench .
docker run --rm aced-bench
```

The data ship with the repository. `bash prepare_dataset.sh` verifies them against `data/SHA256SUMS`.

## Credentials

Running models needs your own API access. The repository contains no keys; the code reads them from your environment
and does not store them.

- **AWS Bedrock (always needed).** Both suites map the agent's free-text requests to the hidden model with a fixed
  parser on Bedrock: Claude Opus 4.8 for ACED-Repair, Claude Opus 4.6 for ACED-Query (which also extracts ACED-Query
  answers). Set `AWS_BEARER_TOKEN_BEDROCK` and `AWS_DEFAULT_REGION`, or use the standard AWS credential chain (for an
  EC2/ECS instance role, also set `BEDROCK_ALLOW_EC2_METADATA=1`; the region defaults to `us-west-2`). Replies are not
  streamed, so `BEDROCK_READ_TIMEOUT` defaults to 900 seconds.
- **Agent models** can run on Bedrock, on any OpenAI-compatible endpoint (`OPENAI_API_KEY`, `OPENAI_BASE_URL`), on a
  local vLLM server (`VLLM_BASE_URL`, ACED-Repair), or on Gemini (`GEMINI_API_KEY`, ACED-Query).

## Agent code runs on your machine

Agents can run Python analysis code on their data.

- **ACED-Repair** runs it in a separate process with a cleaned environment, whose audit hook refuses file access
  outside its working folder, subprocesses, and ctypes file, program and network calls. This is a safeguard against
  accidents, not a security boundary.
- **ACED-Query** (Active+Code and Modular agents) runs it in a separate Python process with no file or network
  restrictions and with your environment variables.

Run untrusted models in a container that holds no credentials beyond the API keys the run needs. World files contain
the answers, so for a public leaderboard do not mount `data/` for the agent's process.

## Development

`bash run_tests.sh` runs the offline tests (no API calls). Pull requests run the checks in `.github/workflows`:
flake8, the tests with a coverage report, a line count, and a check that the three top-level scripts exist.
