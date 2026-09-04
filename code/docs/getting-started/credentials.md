# API keys and provider credentials

The repository can use local Ollama models, Groq-hosted models, OpenAI-hosted models, and Hugging Face model repositories. Credentials are required only for providers used by the selected configuration.

## Environment variables

| Variable | Used when |
|---|---|
| `GROQ_API_KEY` | a Groq-backed prompt or classifier model is selected |
| `OPENAI_API_KEY` | a prompt model name routed to OpenAI is selected |
| `HF_TOKEN` | a private or gated Hugging Face model must be downloaded |

Local Ollama models do not require an API key.

## Local secrets file

The top-level launchers automatically source `${SECRETS_FILE:-<repo>/.secrets.env}` when that file exists.

Create `.secrets.env` manually:

```bash
cat > .secrets.env <<'EOF_SECRETS'
export GROQ_API_KEY=""
export OPENAI_API_KEY=""
export HF_TOKEN=""
EOF_SECRETS
chmod 600 .secrets.env
```

Fill only the variables required by the configured providers. Do not commit `.secrets.env`.

To use a secrets file at another path:

```bash
SECRETS_FILE="$HOME/.config/overtopping/secrets.env" ./run_overtopping_experiments.sh --dry-run
```

For direct Python-module execution, source the file first:

```bash
set -a
. ./.secrets.env
set +a

cd code
python -m studies.overtopping.experiments.run_experiments --dry-run
```

## Provider routing

`core.caching_and_prompting.instruct_model()` routes prompt-model names as follows:

- names beginning with `gpt` or `o` use OpenAI and `OPENAI_API_KEY`;
- `qwen/qwen3.6-27b`, `qwen/qwen3-32b`, and `meta-llama/llama-4-scout-17b-16e-instruct` use Groq and `GROQ_API_KEY`;
- other prompt-model names use local Ollama.

The analyzed Transformer is loaded through the repository model-loading stack. Prompt-provider credentials do not replace local access to the model being causally analyzed.

The jailbreak task can use `BON_JAILBREAK_CLASSIFIER_MODEL` and `BON_JAILBREAK_CLASSIFIER_FALLBACK_MODEL`. Their provider requirements follow the same routing rules.

## Ollama

`setup.sh` attempts to pull:

```text
gemma3:27b
qwen3:4b
```

when the `ollama` executable is available. Additional local models must be installed separately if selected by task or feature configuration.

## Hugging Face access

The top-level experiment/reporting launchers default to:

```bash
HF_HUB_OFFLINE=1
```

This assumes required model files are locally available. To fetch model files intentionally:

```bash
set -a
. ./.secrets.env
set +a

HF_HUB_OFFLINE=0 <command-that-loads-the-model>
```

After the model is available locally, use offline mode for experiment execution.

`HF_MODEL_CACHE_DIR` can point model loading at a specific local Hugging Face cache when supported by the invoking stage.

## Verify credential presence

Check whether variables are set without printing their values:

```bash
python - <<'PY'
import os
for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
    print(f"{name}: {'set' if os.environ.get(name) else 'not set'}")
PY
```

## Credential handling

- Keep credentials out of tracked source files, `data/`, `cache/`, `results/`, command logs, and generated metadata.
- Do not print complete secret values during diagnostics.
- Revoke a credential if it is committed or exposed.
- Credential rotation alone does not invalidate scientific caches unless the provider/model configuration or returned content changes.
