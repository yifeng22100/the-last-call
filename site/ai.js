// "Ask the analyst": Claude answers from this study's verified figures only.
// The visitor's own API key is sent directly to api.anthropic.com from the
// browser and nowhere else. Without a key, a rule-based briefing is offered.
const SDK_URL = "https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.129.0/+esm";
const MODEL = "claude-opus-5-5";
const KEY_NAME = "lastcall.anthropicKey";

const PRESETS = {
  brief: "Write a briefing for a call-centre manager on the current simulator setting: what the rule does, what it would have saved or cost, how sure we are, and the one decision they need to make. Under 220 words.",
  risks: "Act as a sceptical reviewer. What could go wrong if this stopping rule is piloted next month? Cover statistics, operations, customers and fairness. For each risk, say how the pilot plan detects or limits it. Under 280 words.",
  explain: "Explain the method and the main result to a non-technical executive in plain language, without jargon. Under 200 words.",
};

const SYSTEM = `You are the analyst for "The Last Call", a study of when a bank should stop calling clients in a term-deposit telemarketing campaign (UCI Bank Marketing data, Portuguese bank, May 2008 to November 2010; Moro, Cortez & Rita, 2014).

How the study works, in brief:
- A next-call (discrete-time hazard) model estimates the chance that the next call to a client ends in a subscription, using only information known before the call.
- Evaluation is walk-forward: each month from March 2009 is predicted by a model trained only on earlier months. Four candidate models were declared up front; the winner (logistic regression) was chosen on 2009 only; 2010 was held out.
- The deployable rule: keep calling while predicted chance >= cost of a call / value of a subscription. It runs behind safeguards: shadow mode in the first live month, monthly recalibration-in-the-large, and a drift alarm (back to shadow) when last month's odds were off by more than 3x. The safeguards were designed after seeing 2009-2010 results, so for that part those years are not a clean test; a randomised pilot is the real test.
- Every figure in <facts> was computed three independent ways (pandas/scikit-learn, Python standard library, R) and cross-checked.

Rules:
1. Use only the numbers in <facts> and <scenario>. Never introduce outside numbers, benchmarks or costs. If something is not covered, say so plainly.
2. Quote numbers as they appear, rounded sensibly (percentages to one decimal). When you state a result, say which period it comes from.
3. Be candid about limits: one bank, 2008-2010, observational data, the 2010 tie with client scoring, the post-hoc safeguards, illustrative costs.
4. Euro amounts in <scenario> use the visitor's own assumptions; call them assumptions.
5. This is historical analysis, not financial advice.
6. Write in plain English for a business reader. Use short paragraphs or bullets. No preamble.`;

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal, safe Markdown: headings, bullets, bold. Input is escaped first.
function renderMarkdown(md) {
  const lines = escapeHtml(md).split("\n");
  let html = "", list = false;
  const inline = (t) => t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)/);
    if (bullet) { if (!list) { html += "<ul>"; list = true; } html += `<li>${inline(bullet[1])}</li>`; continue; }
    if (list) { html += "</ul>"; list = false; }
    const h = line.match(/^#{1,4}\s+(.*)/);
    if (h) html += `<h4>${inline(h[1])}</h4>`;
    else if (line.trim()) html += `<p>${inline(line)}</p>`;
  }
  if (list) html += "</ul>";
  return html;
}

const store = {
  get() {
    try { return sessionStorage.getItem(KEY_NAME) || localStorage.getItem(KEY_NAME) || ""; } catch { return ""; }
  },
  set(key, remember) {
    try {
      sessionStorage.setItem(KEY_NAME, key);
      if (remember) localStorage.setItem(KEY_NAME, key); else localStorage.removeItem(KEY_NAME);
    } catch { /* storage unavailable: key lives only in the input */ }
  },
};

const signed = (v) => `${v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(1)}`;

function ruleBasedBriefing(s, facts, fmt) {
  const ho = facts.paired_differences.holdout["0.5"].next_call_minus_client_ranking;
  const cut = s.totals.calls - s.calls_made, lost = s.totals.yes - s.subscriptions_kept;
  const verdict = s.net_gain_eur > 0 ? "would have made money" : "would not have made money";
  return [
    "## Briefing (rule-based, no AI)",
    `With a deposit worth €${s.value_eur} and a call costing €${s.cost_eur}, the rule keeps calling a client while the next call has at least a ${fmt.pct1(s.threshold)} chance of success.`,
    `- Over March 2009 to November 2010 it would have made ${fmt.int(s.calls_made)} of the bank's ${fmt.int(s.totals.calls)} calls (${fmt.pct1(s.calls_share)}), cutting ${fmt.int(cut)}.`,
    `- It would have kept ${fmt.int(s.subscriptions_kept)} of ${fmt.int(s.totals.yes)} subscriptions (${fmt.pct1(s.subscriptions_kept_share)}), losing ${fmt.int(lost)}.`,
    `- Under these assumptions it ${verdict}: ${fmt.eur(s.net_gain_eur)} versus the bank's actual calls, or ${fmt.eur(s.net_gain_per_1000_clients_eur)} per 1,000 clients.`,
    `- Caveats: one bank, 2008 to 2010; the safeguards were designed after seeing these years; in 2010 the model ranked no better than plain client scoring (${signed(ho.mean)} points, 95% CI ${signed(ho.ci[0])} to ${signed(ho.ci[1])}).`,
    "**Decision needed:** approve a two-week shadow run followed by a randomised pilot, with net value per client as the primary metric.",
  ].join("\n");
}

export function setupAnalyst({ facts, getScenario, fmt }) {
  const keyIn = document.getElementById("api-key");
  const remember = document.getElementById("remember-key");
  const answer = document.getElementById("answer");
  const copyBtn = document.getElementById("copy-answer");
  const buttons = [...document.querySelectorAll("[data-ask]"), document.querySelector("#ask-form button")];
  keyIn.value = store.get();
  try { remember.checked = !!localStorage.getItem(KEY_NAME); } catch { /* ignore */ }
  let clientPromise = null, lastText = "";

  const setBusy = (busy) => buttons.forEach((b) => { b.disabled = busy; });
  const show = (md, meta = "") => {
    lastText = md;
    answer.innerHTML = renderMarkdown(md) + (meta ? `<p class="meta">${meta}</p>` : "");
    copyBtn.hidden = !md;
  };

  async function ask(question, presetKey) {
    const key = keyIn.value.trim();
    const scenario = getScenario();
    if (!key) {
      if (presetKey === "brief") {
        show(ruleBasedBriefing(scenario, facts, fmt), "Add an Anthropic API key above to have Claude write this instead.");
      } else {
        answer.innerHTML = `<p>Add an Anthropic API key above to ask Claude. You can create one in the Anthropic Console. ` +
          `Without a key, “Brief my manager on this setting” gives a rule-based briefing.</p>`;
      }
      return;
    }
    store.set(key, remember.checked);
    setBusy(true);
    show("Thinking…");
    try {
      clientPromise ??= import(SDK_URL);
      const { default: Anthropic } = await clientPromise;
      const client = new Anthropic({ apiKey: key, dangerouslyAllowBrowser: true });
      const stream = client.beta.messages.stream({
        model: MODEL,
        max_tokens: 4000,
        output_config: { effort: "medium" },
        betas: ["server-side-fallback-2026-07-01"],
        fallbacks: "default",
        system: [
          { type: "text", text: SYSTEM },
          { type: "text", text: `<facts>\n${JSON.stringify(facts)}\n</facts>`, cache_control: { type: "ephemeral" } },
        ],
        messages: [{ role: "user", content: `<scenario>\n${JSON.stringify(scenario)}\n</scenario>\n\n${question}` }],
      });
      let text = "";
      stream.on("text", (delta) => { text += delta; show(text); });
      const msg = await stream.finalMessage();
      if (msg.stop_reason === "refusal") {
        show("Claude declined to answer this request. Try rephrasing the question about the study.");
        return;
      }
      const u = msg.usage || {};
      show(text || "(No text returned.)",
        `Answered by ${escapeHtml(msg.model)} from the study's verified figures · ${u.input_tokens ?? "?"} input / ${u.output_tokens ?? "?"} output tokens` +
        (msg.stop_reason === "max_tokens" ? " · stopped at the length limit" : ""));
    } catch (err) {
      const status = err && err.status;
      const why = status === 401 ? "The API key was rejected. Check it in the Anthropic Console."
        : status === 429 ? "Rate limit reached. Wait a moment and try again."
        : status === 400 ? `The request was rejected: ${escapeHtml(err.message || "bad request")}`
        : status >= 500 ? "Anthropic's API had a temporary problem. Try again shortly."
        : `Could not reach the API (${escapeHtml((err && err.message) || "network error")}).`;
      answer.innerHTML = `<p class="err">${why}</p>`;
      copyBtn.hidden = true;
    } finally {
      setBusy(false);
    }
  }

  document.querySelectorAll("[data-ask]").forEach((b) =>
    b.addEventListener("click", () => ask(PRESETS[b.dataset.ask], b.dataset.ask)));
  document.getElementById("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = document.getElementById("ask-in").value.trim();
    if (q) ask(q, null);
  });
  copyBtn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(lastText); copyBtn.textContent = "Copied"; }
    catch { copyBtn.textContent = "Copy failed"; }
    setTimeout(() => { copyBtn.textContent = "Copy answer"; }, 1600);
  });
}
