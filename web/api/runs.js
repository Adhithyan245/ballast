// POST /api/runs  {question, abstract?}  ->  the same question asked 5 independent times
const B = require("../lib/ballast.js");
const S = require("../lib/server.js");

module.exports = async (req, res) => {
  if (S.cors(req, res)) return;
  if (req.method !== "POST") return res.status(405).json({ error: "Use POST." });
  if (S.limited(req)) return res.status(429).json({ error: "That's a lot of questions in a few minutes. Wait a little and try again." });
  let input;
  try { input = S.readInput(req.body); } catch (e) { return res.status(400).json({ error: "Could not read the request." }); }
  if (input.error) return res.status(400).json({ error: input.error });
  const prompt = B.buildPrompt(input.question, input.abstract);
  try {
    const replies = await Promise.all([1, 2, 3, 4, 5].map(() => S.groq([{ role: "user", content: prompt }])));
    res.status(200).json({ model: replies[0].model, prompt, runs: replies.map((r) => r.text) });
  } catch (e) { S.fail(res, e); }
};
