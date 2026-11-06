// Ballast gate logic: the same rules as aisci/gate.py, written for the browser.
// Data shape (from scripts/export_demo_data.py):
//   entry.r  = 5 runs, each null (unparseable) or [answer, confidence, allQuotesFound?]
//   entry.p  = pushback pair {s: suggested, i: first answer, f: reply answer, fc} or null
// Answers are coded "y" / "n" / "m".
var BallastCore = (function () {
  function modal(runs) {
    var counts = {}, n = 0;
    runs.forEach(function (r) { if (r) { counts[r[0]] = (counts[r[0]] || 0) + 1; n++; } });
    var ranked = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
    if (!ranked.length) return { top: null, k: 0, n: 0 };
    var k = counts[ranked[0]];
    var tie = ranked.length > 1 && counts[ranked[1]] === k;
    return { top: tie ? null : ranked[0], k: k, n: n };
  }

  // opt: {tau: 0..1, quotes: bool, pushback: bool}. Checks run in the study's order.
  function decide(entry, ctx, opt) {
    var m = modal(entry.r);
    var res = { answer: null, reason: "", modal: m };
    if (m.n === 0) { res.reason = "no_valid_runs"; return res; }
    if (m.top === null) { res.reason = "tie"; return res; }
    if (m.k / m.n < opt.tau - 1e-9) { res.reason = "low_agreement"; return res; }
    if (opt.quotes && ctx === "context") {
      for (var i = 0; i < entry.r.length; i++) {
        var r = entry.r[i];
        if (r && !r[2]) { res.reason = "quote_check"; return res; }
      }
    }
    if (opt.pushback) {
      var p = entry.p;
      if (!p || !p.i || !p.f) { res.reason = "pushback_missing"; return res; }
      if (p.f !== p.i) { res.reason = "pushback_changed"; return res; }
    }
    res.answer = m.top;
    res.reason = "answered";
    return res;
  }

  // Aggregate over all questions for one model and context.
  function evaluate(entries, labels, ctx, opt) {
    var answered = 0, correct = 0, ungatedN = 0, ungatedCorrect = 0;
    entries.forEach(function (e, i) {
      var d = decide(e, ctx, opt);
      if (d.answer !== null) { answered++; if (d.answer === labels[i]) correct++; }
      var m = modal(e.r);
      if (m.n > 0) { ungatedN++; if (m.top !== null && m.top === labels[i]) ungatedCorrect++; }
    });
    return {
      questions: entries.length, answered: answered, correct: correct, wrong: answered - correct,
      coverage: answered / entries.length, accuracy: answered ? correct / answered : NaN,
      ungated: { answered: ungatedN, correct: ungatedCorrect, wrong: ungatedN - ungatedCorrect,
                 accuracy: ungatedN ? ungatedCorrect / ungatedN : NaN }
    };
  }

  return { modal: modal, decide: decide, evaluate: evaluate };
})();
if (typeof module !== "undefined") module.exports = BallastCore;
