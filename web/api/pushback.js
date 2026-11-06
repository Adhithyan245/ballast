// POST /api/pushback  {question, abstract?, firstText}  ->  one "Are you sure?" turn
// continuing the conversation of run 1, pushing toward an answer different from its own.
const B = require("../lib/ballast.js");
const S = require("../lib/server.js");

module.exports = async (req, res) => {
  if (S.cors(req, res)) return;
  if (req.method !== "POST") return res.status(405).json({ error: "Use POST." });
  if (S.limited(req)) return res.status(429).json({ error: "That's a lot of questions in a few minutes. Wait a little and try again." });
  let input;
  try { input = S.readInput(req.body); } catch (e) { return res.status(400).json({ error: "Could not read the request." }); }
  if (input.error) return res.status(400).json({ error: input.error });
  const first = B.parse(input.firstText, !!input.abstract);
  if (!first.ok) return res.status(200).json({ skipped: true, reason: "The first answer could not be read, so no pushback was sent." });
  const suggested = B.suggestion(first.answer, input.question);
  const push = B.buildPushback(suggested);
  try {
    const r = await S.groq([
      { role: "user", content: B.buildPrompt(input.question, input.abstract) },
      { role: "assistant", content: input.firstText },
      { role: "user", content: push },
    ]);
    res.status(200).json({ suggested, pushback: push, reply: r.text });
  } catch (e) { S.fail(res, e); }
};
