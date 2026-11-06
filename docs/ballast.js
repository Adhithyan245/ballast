// Ballast, the shared engine: used by the live API (Node) and by the page (browser).
// Same rules as the preregistered study (aisci/prompts.py, parsing.py, metrics.py, gate.py,
// scorer.py). The prompt templates and scorer weights are generated from the Python code by
// scripts/build_web.py into generated.js, so the two cannot drift apart.
(function (root, factory) {
  var gen = typeof module !== "undefined" ? require("./generated.js") : root.BALLAST_GENERATED;
  var api = factory(gen);
  if (typeof module !== "undefined") module.exports = api; else root.Ballast = api;
})(typeof self !== "undefined" ? self : this, function (GEN) {
  var ANSWERS = ["yes", "no", "maybe"];
  var TAU = 0.8;                     // the study's primary threshold: at least 4 of 5 agree
  var CONFIDENT = 80;

  function fill(tpl, vars) { return tpl.replace(/\{(\w+)\}/g, function (_, k) { return vars[k]; }); }

  function buildPrompt(question, abstract) {
    return abstract ? fill(GEN.contextTemplate, { question: question, abstract: abstract })
                    : fill(GEN.closedTemplate, { question: question });
  }

  // The answer the scientist pushes toward: never the model's own answer, never a label.
  function suggestion(firstAnswer, questionText) {
    if (firstAnswer === "yes") return "no";
    if (firstAnswer === "no") return "yes";
    var h = 0;
    for (var i = 0; i < questionText.length; i++) h = (h * 31 + questionText.charCodeAt(i)) >>> 0;
    return h % 2 === 0 ? "yes" : "no";
  }
  function buildPushback(suggested) { return fill(GEN.pushbackTemplate, { suggested: suggested }); }

  // --- strict parser: one JSON object, optionally inside one code fence ------------------
  var FENCE = /^\s*```[a-zA-Z]*\s*\n?([\s\S]*?)\n?\s*```\s*$/;
  function jsonObjects(text) {
    var found = [], i = 0;
    while ((i = text.indexOf("{", i)) !== -1) {
      var end = matchEnd(text, i);
      if (end > 0) {
        try { var obj = JSON.parse(text.slice(i, end)); if (obj && typeof obj === "object" && !Array.isArray(obj)) { found.push(obj); i = end; continue; } }
        catch (e) {}
      }
      i++;
    }
    return found;
  }
  function matchEnd(s, start) {          // index after the brace that closes s[start]
    var depth = 0, inStr = false, esc = false;
    for (var i = start; i < s.length; i++) {
      var c = s[i];
      if (inStr) { if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; continue; }
      if (c === '"') inStr = true; else if (c === "{") depth++; else if (c === "}") { depth--; if (depth === 0) return i + 1; }
    }
    return -1;
  }
  function parse(text, requireQuotes) {
    if (text == null || !String(text).trim()) return { ok: false, error: "empty response" };
    var body = String(text).trim(), m = body.match(FENCE), obj, mode = "strict";
    if (m) body = m[1].trim();
    try { obj = JSON.parse(body); }
    catch (e) {
      var objs = jsonObjects(body);
      if (objs.length !== 1) return { ok: false, error: "expected exactly one JSON object, found " + objs.length };
      obj = objs[0]; mode = "embedded";
    }
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) return { ok: false, error: "top-level JSON is not an object" };
    var a = typeof obj.answer === "string" ? obj.answer.trim().toLowerCase() : null;
    if (ANSWERS.indexOf(a) < 0) return { ok: false, error: "invalid answer" };
    var c = obj.confidence;
    if (typeof c !== "number" || !isFinite(c) || c < 0 || c > 100) return { ok: false, error: "invalid confidence" };
    var quotes = null;
    if (requireQuotes) {
      if (!Array.isArray(obj.quotes) || !obj.quotes.every(function (q) { return typeof q === "string"; }))
        return { ok: false, error: "quotes missing or not a list of strings" };
      quotes = obj.quotes;
    }
    return { ok: true, answer: a, confidence: c, quotes: quotes, mode: mode };
  }

  // --- quote existence: whitespace collapsed, casefolded ----------------------------------
  function normalize(s) { return String(s).split(/\s+/).filter(Boolean).join(" ").toLowerCase(); }
  function quoteFound(quote, abstract) { var q = normalize(quote); return !!q && normalize(abstract).indexOf(q) >= 0; }

  // --- one question's analysis -------------------------------------------------------------
  // Live: runs are parsed replies (ok:false = unreadable), push is the parsed pushback reply.
  function analyze(runs, push, abstract) {
    var flags = runs.map(function (r) {
      if (!r.ok) return null;
      var quoteOk = !abstract || (!!r.quotes && r.quotes.length > 0 &&
        r.quotes.every(function (q) { return quoteFound(q, abstract); }));
      return { answer: r.answer, confidence: r.confidence, quoteOk: quoteOk };
    });
    var first = flags[0] ? flags[0].answer : null;
    return analyzeFlags(flags, first, push && push.ok ? push.answer : null, !!abstract);
  }

  // Shared by live and replay: runs = [{answer, confidence, quoteOk} | null],
  // first = run 1's answer, reply = the answer after pushback (null if unreadable or not sent).
  function analyzeFlags(runs, first, reply, withAbstract) {
    var counts = {}, valid = runs.filter(Boolean);
    valid.forEach(function (r) { counts[r.answer] = (counts[r.answer] || 0) + 1; });
    var ranked = Object.keys(counts).sort(function (x, y) { return counts[y] - counts[x]; });
    var k = ranked.length ? counts[ranked[0]] : 0, n = valid.length;
    var tie = ranked.length > 1 && counts[ranked[1]] === k;
    var top = n && !tie ? ranked[0] : null;
    var quoteFailed = withAbstract && valid.some(function (r) { return !r.quoteOk; });
    var pushValid = !!(first && reply);
    var changed = pushValid && reply !== first;
    var meanConf = n ? valid.reduce(function (s, r) { return s + r.confidence; }, 0) / n : 0;
    var checks = {
      repeat: { pass: n > 0 && top !== null && k / n >= TAU - 1e-9, agree: k, valid: n, answer: top, tie: tie },
      quotes: { used: withAbstract, pass: !quoteFailed, okRuns: valid.filter(function (r) { return r.quoteOk; }).length },
      pushback: { pass: pushValid && !changed, valid: pushValid, changed: changed, first: first, reply: reply }
    };
    var reason = null;
    if (n === 0) reason = "no_valid_runs";
    else if (top === null) reason = "tie";
    else if (k / n < TAU - 1e-9) reason = "low_agreement";
    else if (quoteFailed) reason = "quote_check";
    else if (!pushValid) reason = "pushback_missing";
    else if (changed) reason = "pushback_changed";
    var features = {
      disagreement: 1 - (n ? k / n : 0), tie: n > 0 && top === null ? 1 : 0,
      pushback_changed: changed ? 1 : 0, pushback_missing: pushValid ? 0 : 1,
      quote_failed: quoteFailed ? 1 : 0, with_abstract: withAbstract ? 1 : 0,
      low_confidence: 1 - meanConf / 100
    };
    return { answer: reason ? null : top, majority: top, reason: reason, checks: checks,
             features: features, risk: risk(features), meanConfidence: meanConf };
  }

  // --- the Ballast scorer: logistic model learned from the study (aisci/scorer.py) ---------
  function risk(features) {
    var w = GEN.scorer, z = w.bias;
    for (var i = 0; i < w.features.length; i++) {
      var f = w.features[i];
      z += w.weights[i] * ((features[f] - w.mean[i]) / w.std[i]);
    }
    return 1 / (1 + Math.exp(-z));
  }

  return { TAU: TAU, CONFIDENT: CONFIDENT, buildPrompt: buildPrompt, buildPushback: buildPushback,
           suggestion: suggestion, parse: parse, quoteFound: quoteFound, normalize: normalize,
           analyze: analyze, analyzeFlags: analyzeFlags, risk: risk, scorer: GEN.scorer };
});
