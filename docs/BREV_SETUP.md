# Self-hosted open-weight model on NVIDIA Brev (fifth family)

The fifth family is an open-weight model (planned: Qwen3, about 32B) that we serve ourselves
with [vLLM](https://docs.vllm.ai) on a GPU rented through NVIDIA Brev, paid for with Brev
credits. vLLM exposes an OpenAI-style `/v1/chat/completions` endpoint, which
`aisci/providers/openai_compat.py::VLLMProvider` calls.

Why self-host: the exact weights (Hugging Face revision hash), server version and GPU are
pinned and recorded, so the model cannot change between runs. There are no rate limits, and
temperature can be set.

**A running instance spends credits every hour.** Nothing here is launched automatically.
The GPU type and hourly price are approved by the account owner before any instance starts.

## 1. Choose and approve the GPU

- Need: one GPU with 80 GB memory (A100 80GB or H100 80GB) for a ~32B model in BF16.
- Read the hourly price in the Brev console, and record it in `results/vllm_server.json` (step 5).
- Budget check: GPU-hours × price must stay well under the $75 credit balance. The pilot
  (400 calls) is run first and timed, and the main-study cost is estimated from that.

## 2. Pin the model

Pick the model at run time from Hugging Face and record its commit hash:

```bash
pip install -U huggingface_hub
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/<MODEL>').sha)"
```

## 3. Start vLLM on the instance

```bash
pip install vllm==<PINNED_VERSION>
export VLLM_API_KEY=<long random string; also put it in your local .env>
vllm serve Qwen/<MODEL> --revision <HF_SHA> \
  --api-key "$VLLM_API_KEY" --port 8000 \
  --max-model-len 16384 --reasoning-parser qwen3 --seed 0
```

- Default sampling = the model's own `generation_config.json`, which vLLM applies by default.
  That is the self-hosted equivalent of "provider default" for the API models. The t0
  condition sends `temperature: 0`.
- `--reasoning-parser qwen3` moves the model's thinking into `reasoning_content`, so only the
  final answer is parsed. The thinking text is still saved in the raw response.
- Do not expose port 8000 publicly. Reach it through an SSH tunnel or Brev port forwarding
  (check `brev port-forward --help` for the current syntax), so it shows up as `localhost:8000`.

## 4. Run from this machine

```bash
# .env: VLLM_BASE_URL=http://localhost:8000/v1 and VLLM_API_KEY=<same string>
python -m aisci smoke vllm
python -m aisci run pilot --families vllm
```

`study_models.json` sets `"workers"` for vLLM, so calls are sent in parallel and the GPU is
used efficiently. The API providers always run one call at a time.

## 5. Record, then STOP the instance

Write `results/vllm_server.json` with: GPU type, hourly price, `nvidia-smi` output, vLLM
version, model id and HF revision, the full `vllm serve` command, and start/stop times (UTC).

Then stop the instance in the Brev console (or with `brev stop <instance>`) as soon as the
runs finish. A stopped instance has no compute charges, only small storage costs. Deleting
it costs nothing.
