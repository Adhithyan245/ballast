// Server helpers for the live API: input checks, CORS, a small per-IP limit, and Groq calls.
// The Groq key lives only in the server environment (GROQ_API_KEY); it never reaches the page.
const MODEL = "openai/gpt-oss-120b";          // the study's gpt-oss model, on Groq's free plan
const MAX_QUESTION = 400;
const MAX_ABSTRACT = 6000;
const ALLOWED_ORIGINS = ["https://adhithyan245.github.io"];
const WINDOW_MS = 10 * 60 * 1000, MAX_PER_WINDOW = 20;
const hits = new Map();                       // best effort: per server instance

function cors(req, res) {
  const origin = req.headers.origin;
  if (origin && ALLOWED_ORIGINS.includes(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  }
  if (req.method === "OPTIONS") { res.status(204).end(); return true; }
  return false;
}

function limited(req) {
  const ip = String(req.headers["x-forwarded-for"] || "").split(",")[0].trim() || "unknown";
  const now = Date.now();
  const list = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  list.push(now);
  hits.set(ip, list);
  return list.length > MAX_PER_WINDOW;
}

function readInput(body) {
  const b = typeof body === "string" ? JSON.parse(body || "{}") : body || {};
  const question = String(b.question || "").trim();
  const abstract = String(b.abstract || "").trim();
  if (!question) return { error: "Write a question first." };
  if (question.length > MAX_QUESTION) return { error: `Keep the question under ${MAX_QUESTION} characters.` };
  if (abstract.length > MAX_ABSTRACT) return { error: `Paste at most ${MAX_ABSTRACT} characters of the abstract.` };
  return { question, abstract: abstract || null, firstText: b.firstText ? String(b.firstText).slice(0, 8000) : null };
}

async function groq(messages) {
  const key = process.env.GROQ_API_KEY;
  if (!key) throw Object.assign(new Error("The live engine is not configured yet."), { status: 503 });
  for (let attempt = 1; attempt <= 3; attempt++) {
    const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: MODEL, messages }),   // default sampling, as in the study
    });
    if (r.ok) {
      const j = await r.json();
      return { text: j.choices?.[0]?.message?.content ?? null, model: j.model || MODEL };
    }
    if (r.status === 429) {
      const msg = await r.text();
      if (/per day/i.test(msg)) throw Object.assign(new Error("Today's free quota for the live engine is used up. Try again tomorrow, or replay a study case."), { status: 429 });
      const wait = Math.min(8, Number(r.headers.get("retry-after")) || attempt * 2);
      await new Promise((ok) => setTimeout(ok, wait * 1000));
      continue;
    }
    if (r.status >= 500 && attempt < 3) { await new Promise((ok) => setTimeout(ok, attempt * 1000)); continue; }
    throw Object.assign(new Error(`The model service returned an error (${r.status}).`), { status: 502 });
  }
  throw Object.assign(new Error("The model service is busy. Try again in a minute."), { status: 503 });
}

function fail(res, e) {
  res.status(e.status || 500).json({ error: e.message || "Something went wrong." });
}

module.exports = { MODEL, cors, limited, readInput, groq, fail };
