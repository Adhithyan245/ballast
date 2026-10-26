# Getting the keys and logins

Put every key in `.env` (git-ignored). Never paste a key into chat, an issue, or a commit.
After each step, run `python -m aisci check`. It reports READY / NOT READY per provider,
never prints key values, and uses no quota.

## 1. Claude (subscription login, no key)

```bash
claude update
claude auth login
claude auth status
```

`auth status` should show `"loggedIn": true`. These are the same limits your interactive
Claude use draws on.

## 2. Gemini: Google AI Studio free tier

1. Open https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key**. Use a new project if asked, and don't enable billing (that keeps it on the free tier).
3. Copy the key into `.env` as `GEMINI_API_KEY=...`.
4. Note the free-tier limits AI Studio shows for your project (RPM, TPM, RPD per model). They
   are recorded in `study_models.json` at run time.

## 3. Groq: free plan

1. Open https://console.groq.com/keys and sign in.
2. Click **Create API Key**, name it `aisci`, and copy it into `.env` as `GROQ_API_KEY=...`.
3. Your plan's limits are listed at https://console.groq.com/settings/limits. Each response also
   reports them in its headers, and the cache saves those headers.

## 4. Mistral: free Experiment plan

1. Open https://console.mistral.ai and sign in.
2. Choose the **Experiment** (free) plan. It requires phone verification and **opting into
   data training**. That's acceptable here because the study sends only public PubMedQA text.
3. Go to **API Keys**, then **Create new key**, and copy it into `.env` as `MISTRAL_API_KEY=...`.
4. Note the limits shown for your workspace.

## 5. Self-hosted vLLM (Brev): later, not needed for the first checks

`VLLM_API_KEY` is a password you invent for your own vLLM server. Generate one and append it
to `.env` without displaying it:

```bash
python -c "import secrets; open('.env','a').write('\nVLLM_API_KEY=' + secrets.token_urlsafe(32) + '\n')"
```

Use the same value with `vllm serve --api-key` on the instance (docs/BREV_SETUP.md).
Remove the empty `VLLM_API_KEY=` line from the template so only one remains.
