# Setup

## Environment

Python 3.12. pgmpy needs torch, and the CPU wheel is enough, so install it first:

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Or use Docker (the image runs the tests by default):

```bash
docker build -t aced-bench .
docker run --rm aced-bench
```

## API keys

Use your own API keys; they are read from your environment.

- **AWS Bedrock (always needed).** The request parser runs on Bedrock (Claude Opus 4.8 for ACED-Repair, Claude
  Opus 4.6 for ACED-Query). Set `AWS_BEARER_TOKEN_BEDROCK` and `AWS_DEFAULT_REGION`, or use the standard AWS
  credential chain.
- **Agent models** can run on Bedrock, an OpenAI-compatible endpoint (`OPENAI_API_KEY`, `OPENAI_BASE_URL`), a local
  vLLM server (`VLLM_BASE_URL`, ACED-Repair), or Gemini (`GEMINI_API_KEY`, ACED-Query).

## Agent code runs on your machine

Agents can run Python code on their data. ACED-Repair runs it in a restricted process (a safeguard against accidents,
not a security boundary); ACED-Query runs it without restrictions. Run untrusted models in a container with no
credentials beyond the keys the run needs. World files contain the answers, so for a public leaderboard do not mount
`data/` for the agent's process.
