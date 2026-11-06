// One live end-to-end run of the API handlers against Groq (uses GROQ_API_KEY from ../../.env).
//   node web/tests/live_smoke.js
const fs = require("fs");
const path = require("path");
const env = fs.readFileSync(path.join(__dirname, "..", "..", ".env"), "utf8");
const m = env.match(/^GROQ_API_KEY=(.*)$/m);
process.env.GROQ_API_KEY = m ? m[1].trim() : "";
const B = require("../lib/ballast.js");
const runs = require("../api/runs.js");
const pushback = require("../api/pushback.js");

function call(handler, body) {
  return new Promise((resolve) => {
    const res = { headers: {}, statusCode: 200, setHeader(k, v) { this.headers[k] = v; },
      status(c) { this.statusCode = c; return this; }, json(o) { resolve({ status: this.statusCode, body: o }); }, end() { resolve({ status: this.statusCode }); } };
    handler({ method: "POST", headers: { "x-forwarded-for": "127.0.0.1" }, body }, res);
  });
}

(async () => {
  const study = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "study-data.json"), "utf8"));
  const q = study.questions[0];
  const abstract = q.abstract.map((p) => (p[0] ? p[0] + ": " + p[1] : p[1])).join("\n\n");
  const t0 = Date.now();
  const r = await call(runs, { question: q.q, abstract });
  console.log("runs:", r.status, r.body.error || `${r.body.runs.length} replies from ${r.body.model} in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  if (r.status !== 200) process.exit(1);
  const parsed = r.body.runs.map((t) => B.parse(t, true));
  const t1 = Date.now();
  const p = await call(pushback, { question: q.q, abstract, firstText: r.body.runs[0] });
  console.log("pushback:", p.status, p.body.error || `suggested "${p.body.suggested}" in ${((Date.now() - t1) / 1000).toFixed(1)}s`);
  const a = B.analyze(parsed, p.body.reply ? B.parse(p.body.reply, true) : null, abstract);
  console.log("parsed:", parsed.map((x) => (x.ok ? x.answer + "/" + x.confidence : "unreadable")).join(" "));
  console.log("verdict:", a.answer ? "pass: " + a.answer : "hand back: " + a.reason, "| risk", a.risk.toFixed(2), "| expert label", q.label);
  const bad = await call(runs, { question: "" });
  console.log("empty question rejected:", bad.status === 400);
})();
