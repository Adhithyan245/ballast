// The JS engine must behave exactly like the Python study code.
//   node web/tests/parity.js
const fs = require("fs");
const path = require("path");
const B = require("../lib/ballast.js");
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "parse_fixtures.json"), "utf8"));
const pf = JSON.parse(fs.readFileSync(path.join(__dirname, "prompt_fixtures.json"), "utf8"));
let bad = 0;
for (const f of fx) {
  const r = B.parse(f.t, f.q);
  const same = r.ok === f.ok && (!f.ok || (r.answer === f.a && r.confidence === f.c &&
    (f.n === null ? r.quotes === null : r.quotes && r.quotes.length === f.n)));
  if (!same && bad++ < 5) console.log("PARSE MISMATCH", JSON.stringify({ py: f, js: r }).slice(0, 300));
}
console.log(`${fx.length - bad}/${fx.length} study replies parse identically in JS and Python`);
let pbad = 0;
for (const p of pf) {
  if (B.buildPrompt(p.question, null) !== p.closed) pbad++;
  if (B.buildPrompt(p.question, p.abstract) !== p.context) pbad++;
  if (B.buildPushback("no") !== p.push_no) pbad++;
}
console.log(`${pf.length * 3 - pbad}/${pf.length * 3} prompts identical to the study's`);
// quote rule and suggestion rule
const ok = B.quoteFound("The   drug\nWORKED.", "x. The drug worked. y") && !B.quoteFound("  ", "abc") &&
  B.suggestion("yes", "q") === "no" && B.suggestion("no", "q") === "yes" &&
  ["yes", "no"].includes(B.suggestion("maybe", "q"));
console.log("quote and suggestion rules:", ok ? "ok" : "FAILED");
// scorer: risk in (0,1), more disagreement -> more risk
const base = { disagreement: 0, tie: 0, pushback_changed: 0, pushback_missing: 0, quote_failed: 0, with_abstract: 1, low_confidence: 0.1 };
const worse = Object.assign({}, base, { disagreement: 0.4, pushback_changed: 1, low_confidence: 0.4 });
const sOk = B.risk(base) > 0 && B.risk(worse) < 1 && B.risk(worse) > B.risk(base);
console.log("scorer monotonic:", sOk ? "ok" : "FAILED", B.risk(base).toFixed(3), B.risk(worse).toFixed(3));
process.exit(bad || pbad || !ok || !sOk ? 1 : 0);
