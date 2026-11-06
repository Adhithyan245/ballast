// Node check: the page's gate logic must reproduce the study's gate numbers exactly.
//   node demo/check_core.js
const fs = require("fs");
const path = require("path");
const core = require("./ballast_core.js");
const root = path.join(__dirname, "..");
const data = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const summary = JSON.parse(fs.readFileSync(path.join(root, "results/analysis/summary.json"), "utf8"));
const labels = data.questions.map((q) => q.label);
let ok = 0, bad = 0;
function same(what, a, b) {
  if (a === b) ok++; else { bad++; console.log(`MISMATCH ${what}: page ${a} vs study ${b}`); }
}
for (const m of data.models) {
  for (const ctx of ["closed", "context"]) {
    const g = summary.models[m.key].gates[ctx];
    const cons = core.evaluate(m.ctx[ctx], labels, ctx, { tau: 0.8, quotes: true, pushback: false });
    const comb = core.evaluate(m.ctx[ctx], labels, ctx, { tau: 0.8, quotes: true, pushback: true });
    same(`${m.name} ${ctx} consistency answered`, cons.answered, g.consistency_gate.answered);
    same(`${m.name} ${ctx} consistency correct`, cons.correct, g.consistency_gate.correct);
    same(`${m.name} ${ctx} combined answered`, comb.answered, g.combined_gate.answered);
    same(`${m.name} ${ctx} combined correct`, comb.correct, g.combined_gate.correct);
    same(`${m.name} ${ctx} ungated correct`, cons.ungated.correct, g.ungated_majority.correct);
    same(`${m.name} ${ctx} ungated answered`, cons.ungated.answered, g.ungated_majority.answered);
    for (const tau of [0, 0.6, 1.0]) {
      const curve = summary.models[m.key].gate_curves[ctx];
      for (const [gate, pb] of [["consistency", false], ["combined", true]]) {
        const pt = curve[gate].find((p) => Math.abs(p.threshold - tau) < 1e-9);
        const r = core.evaluate(m.ctx[ctx], labels, ctx, { tau, quotes: true, pushback: pb });
        same(`${m.name} ${ctx} ${gate} tau=${tau} answered`, r.answered, pt.answered);
        same(`${m.name} ${ctx} ${gate} tau=${tau} correct`, r.correct, pt.correct);
      }
    }
  }
}
console.log(`${ok}/${ok + bad} gate numbers on the page match results/analysis/summary.json`);
process.exit(bad ? 1 : 0);
