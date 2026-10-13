// The Last Call - page logic. Every number shown comes from data/*.json,
// which the build writes and the verifiers check; nothing is typed by hand.
import { barChart, hbarChart, lineChart, legend, showTip, hideTip } from "./charts.js";
import { setupAnalyst } from "./ai.js";

const load = (name) => fetch(`data/${name}.json`).then((r) => {
  if (!r.ok) throw new Error(`data/${name}.json: ${r.status}`);
  return r.json();
});

const fmt = {
  pct0: (v) => `${(v * 100).toFixed(0)}%`,
  pct1: (v) => `${(v * 100).toFixed(1)}%`,
  pct2: (v) => `${(v * 100).toFixed(2)}%`,
  num1: (v) => v.toFixed(1),
  num2: (v) => v.toFixed(2),
  num3: (v) => v.toFixed(3),
  int: (v) => Math.round(v).toLocaleString("en-US"),
  eur: (v) => `${v < 0 ? "−" : ""}€${Math.abs(Math.round(v)).toLocaleString("en-US")}`,
};
const pctCI = (ci) => `${fmt.pct1(ci[0])} to ${fmt.pct1(ci[1])}`;

function get(obj, path) {
  return path.split("/").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

function bindFacts(src) {
  for (const el of document.querySelectorAll("[data-fact]")) {
    const v = get(src, el.dataset.fact);
    if (v === undefined || v === null) {
      el.textContent = "n/a";
      console.error("missing fact", el.dataset.fact);
      continue;
    }
    const f = fmt[el.dataset.format];
    el.textContent = typeof v === "number" ? (f ? f(v) : fmt.num3(v)) : String(v);
  }
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const periodLabel = (p) => `${MONTHS[+p.slice(5, 7) - 1]} ${p.slice(0, 4)}`;

// ------------------------------------------------------------------ hero
function hero(facts) {
  const rows = facts.per_call;
  const readout = document.getElementById("strip-readout");
  const describe = (i) => {
    const r = rows[i];
    const which = r.call === "10+" ? "Calls number 10 and beyond" : `Call ${r.call}`;
    readout.innerHTML = `<b>${which}</b> was made ${fmt.int(r.calls_made)} times and ${fmt.int(r.yes)} ended in a yes: ` +
      `<b>${fmt.pct1(r.rate)}</b> (95% CI ${pctCI(r.ci)}). By the end of ${r.call === "10+" ? "all calls" : `call ${r.call}`}, ` +
      `${fmt.pct1(r.cum_share_of_yes)} of all ${fmt.int(facts.dataset.subscribed)} subscriptions had happened.`;
  };
  barChart(document.getElementById("hero-chart"), {
    yMax: 0.07, yFormat: (v) => `${(v * 100).toFixed(0)}%`,
    bars: rows.map((r) => ({
      label: r.call, value: r.rate, lo: r.ci[0], hi: r.ci[1],
      aria: `Call ${r.call}: ${fmt.pct1(r.rate)}`,
      tip: `<b>Call ${r.call}</b><br>${fmt.pct1(r.rate)} succeeded<br>95% CI ${pctCI(r.ci)}<br>${fmt.int(r.calls_made)} calls made`,
    })),
    onFocus: describe,
  });
  describe(1);
}

// -------------------------------------------------------------- findings
function aucChart(facts) {
  const a = facts.auc;
  hbarChart(document.getElementById("auc-chart"), {
    xMin: 0.5, xMax: 1, format: (v) => v.toFixed(2),
    rows: [
      { label: "All 20 columns, shuffled split", value: a.typical_all_inputs_random_split,
        tip: "Includes <b>duration</b>, known only after the call ends" },
      { label: "Without call duration, shuffled", value: a.without_duration_random_split,
        tip: "Still includes total calls in the campaign and last-contact timing" },
      { label: "Pre-call information only, shuffled", value: a.precall_random_split,
        tip: "Client profile, previous campaign, economic indicators" },
      { label: "Pre-call only, tested on future months", value: a.precall_walk_forward_gradient_boosting, emphasis: true,
        tip: `Trained on earlier months, scored on the next. Mean within-month AUC ${fmt.num3(a.precall_walk_forward_gradient_boosting_mean_within_month)}.` },
    ],
  });
}

function countingChart(facts) {
  const per = facts.per_call.map((r, i) => ({ x: i + 1, y: r.rate, lo: r.ci[0], hi: r.ci[1], n: r.calls_made }));
  const naive = facts.naive_by_campaign.map((r, i) => ({ x: i + 1, y: r.conversion, n: r.clients }));
  lineChart(document.getElementById("counting-chart"), {
    xDomain: [0.7, 10.3], yDomain: [0, 0.14],
    xTicks: per.map((p) => ({ v: p.x, label: p.x === 10 ? "10+" : String(p.x) })),
    yFormat: (v) => `${(v * 100).toFixed(0)}%`,
    tipX: (x) => (x === 10 ? "10 or more" : `Call / total calls = ${x}`),
    tipY: (y, p, s) => `${fmt.pct1(y)} <span class="ci">(${fmt.int(p.n)} ${s.unit})</span>`,
    series: [
      { name: "Common view", unit: "clients", color: "--series-2", points: naive, markers: true },
      { name: "Success rate of that call", unit: "calls", color: "--series-1", points: per, markers: true, band: true },
    ],
  });
  legend(document.getElementById("counting-legend"), [
    { color: "--series-2", label: "Common view: conversion of clients called k times in total" },
    { color: "--series-1", label: "Correct view: success rate of call k among clients still being called (band: 95% CI)" },
  ]);
}

function driftCharts(monthly) {
  const maxIdx = 30;
  const byIdx = new Map(monthly.map((m) => [m.month_idx, m]));
  const series = (key, from = 0) => Array.from({ length: maxIdx + 1 }, (_, i) => {
    const m = byIdx.get(i);
    return { x: i, y: m && i >= from && m[key] != null ? m[key] : null, n: m ? m.clients : 0, period: m ? m.period : null };
  });
  const xTicks = [{ v: 0, label: "May 2008", anchor: "start" }, { v: 8, label: "2009" }, { v: 20, label: "2010" },
    { v: 30, label: "Nov 2010", anchor: "end" }];
  const tipX = (x) => {
    const m = byIdx.get(x);
    return m ? periodLabel(m.period) : "No calls this month";
  };
  lineChart(document.getElementById("drift-chart"), {
    xDomain: [0, maxIdx], yDomain: [0, 0.7], xTicks, maxGap: 1, tipX,
    yFormat: (v) => `${(v * 100).toFixed(0)}%`,
    tipY: (y, p, s) => s.name === "Actual" ? `${fmt.pct1(y)} <span class="ci">(${fmt.int(p.n)} clients)</span>` : fmt.pct1(y),
    series: [
      { name: "Actual", color: "--series-1", points: series("conversion"), markers: true },
      { name: "Model frozen at end of 2008", color: "--series-2", points: series("f_gradient_boosting", 10), markers: true },
    ],
  });
  legend(document.getElementById("drift-legend"), [
    { color: "--series-1", label: "Actual share who subscribed" },
    { color: "--series-2", label: "Predicted by a model trained on 2008 and never updated" },
  ]);
  lineChart(document.getElementById("euribor-chart"), {
    xDomain: [0, maxIdx], yDomain: [0, 5.5], xTicks, maxGap: 1, tipX,
    yFormat: (v) => `${v}`, tipY: (y) => `${y.toFixed(2)}%`,
    margin: { l: 46, r: 14, t: 8, b: 26 },
    series: [{ name: "3-month Euribor", color: "--series-3", points: series("euribor"), markers: true }],
  });
}

// ------------------------------------------------------------- simulator
function interpGrid(grid, t) {
  const th = grid.threshold;
  if (t <= th[0]) return { calls: grid.calls[0], yes: grid.yes[0] };
  if (t >= th[th.length - 1]) return { calls: grid.calls[th.length - 1], yes: grid.yes[th.length - 1] };
  let i = th.findIndex((v) => v >= t);
  if (th[i] === t) return { calls: grid.calls[i], yes: grid.yes[i] };
  const w = (t - th[i - 1]) / (th[i] - th[i - 1]);
  return { calls: grid.calls[i - 1] + w * (grid.calls[i] - grid.calls[i - 1]),
           yes: grid.yes[i - 1] + w * (grid.yes[i] - grid.yes[i - 1]) };
}

function simulator(facts, curves) {
  const grid = curves.guarded_threshold_grid;
  const T = curves.totals;
  const els = Object.fromEntries(["value-in", "value-range", "cost-in", "cost-range", "sim-rule", "sim-readouts"]
    .map((id) => [id, document.getElementById(id)]));
  const frontier = (pts) => [{ x: 0, y: 0 }, ...pts];
  const lastCall = grid.threshold.map((t, i) => ({ x: grid.calls[i] / T.calls, y: grid.yes[i] / T.yes, t }))
    .sort((a, b) => a.x - b.x);
  const client = frontier(curves.client_ranking_logistic.calls.map((c, i) =>
    ({ x: c / T.calls, y: curves.client_ranking_logistic.yes[i] / T.yes })));
  const cap = frontier(curves.call_cap.cap.map((n, i) =>
    ({ x: curves.call_cap.calls[i] / T.calls, y: curves.call_cap.yes[i] / T.yes, cap: n })));
  const state = {};

  function scenario() {
    const V = Math.max(+els["value-in"].value || 0, 0.01), C = Math.max(+els["cost-in"].value || 0, 0.01);
    const t = C / V;
    const op = interpGrid(grid, t);
    const gain = C * (T.calls - op.calls) - V * (T.yes - op.yes);
    let best = { gain: -Infinity };
    grid.threshold.forEach((th, i) => {
      const g = C * (T.calls - grid.calls[i]) - V * (T.yes - grid.yes[i]);
      if (g > best.gain) best = { gain: g, t: th };
    });
    if (best.gain < 0) best = { gain: 0, t: null };  // calling everyone is the hindsight best
    return { value_eur: V, cost_eur: C, threshold: t, calls_made: op.calls, calls_share: op.calls / T.calls,
      subscriptions_kept: op.yes, subscriptions_kept_share: op.yes / T.yes, net_gain_eur: gain,
      net_gain_per_1000_clients_eur: (gain / T.clients) * 1000, hindsight_best: best,
      period: "March 2009 to November 2010 (out-of-sample)", totals: T };
  }

  function render() {
    const s = scenario();
    Object.assign(state, s);
    els["sim-rule"].innerHTML = `Rule: keep calling a client while the next call has at least a <b>${fmt.pct1(s.threshold)}</b> chance of a yes ` +
      `(€${s.cost_eur} ÷ €${s.value_eur}).`;
    const best = s.hindsight_best.t === null ? "Call everyone (no cut pays off)"
      : `${fmt.eur(s.hindsight_best.gain)} at a ${fmt.pct1(s.hindsight_best.t)} threshold`;
    els["sim-readouts"].innerHTML = [
      ["Calls made", fmt.int(s.calls_made), `${fmt.pct1(s.calls_share)} of the bank's ${fmt.int(T.calls)}`],
      ["Subscriptions kept", fmt.int(s.subscriptions_kept), `${fmt.pct1(s.subscriptions_kept_share)} of ${fmt.int(T.yes)}`],
      ["Net gain vs. the bank's calls", fmt.eur(s.net_gain_eur), `${fmt.eur(s.net_gain_per_1000_clients_eur)} per 1,000 clients`],
      ["Best possible in hindsight", s.hindsight_best.t === null ? "€0" : fmt.eur(s.hindsight_best.gain), best === "Call everyone (no cut pays off)" ? best : "threshold picked after seeing outcomes"],
    ].map(([k, v, sub]) => `<div><dt>${k}</dt><dd>${v}<small>${sub}</small></dd></div>`).join("");
    lineChart(document.getElementById("frontier-chart"), {
      xDomain: [0, 1], yDomain: [0, 1], diagonal: true,
      xTicks: [0, 0.25, 0.5, 0.75, 1].map((v) => ({ v, label: `${v * 100}%`, anchor: v === 0 ? "start" : v === 1 ? "end" : "middle" })),
      yFormat: (v) => `${Math.round(v * 100)}%`,
      tipX: (x) => `${fmt.pct0(x)} of the bank's calls made`,
      tipY: (y, p, ser) => fmt.pct1(y) + " of subscriptions" + (ser.name.startsWith("Max") && p.cap ? "" : ""),
      series: [
        { name: "Max-N-calls rule", color: "--series-3", points: cap, markers: true, interp: true, width: 1.5 },
        { name: "Client scoring", color: "--series-2", points: client, interp: true },
        { name: "The Last Call", color: "--series-1", points: lastCall, interp: true, width: 2.5 },
      ],
      marker: { x: s.calls_share, y: s.subscriptions_kept_share, color: "--series-1" },
    });
  }

  const link = (num, range) => {
    els[num].addEventListener("input", () => { els[range].value = els[num].value; render(); });
    els[range].addEventListener("input", () => { els[num].value = els[range].value; render(); });
  };
  link("value-in", "value-range");
  link("cost-in", "cost-range");
  document.getElementById("sim-form").addEventListener("submit", (e) => e.preventDefault());
  legend(document.getElementById("frontier-legend"), [
    { color: "--series-1", label: "The Last Call (with safeguards)" },
    { color: "--series-2", label: "Client scoring: rank clients, make all the bank's calls to the top ones" },
    { color: "--series-3", label: "Max-N-calls rule (N = 1, 2, 3…)", kind: "dot" },
    { color: "--neutral", label: "Random" },
  ]);
  render();
  return () => ({ ...state });
}

// ---------------------------------------------------------------- tables
function compareTable(facts) {
  const sel = facts.model_selection.selected;
  const W = [
    ["validation", "2009 (model chosen here)", facts.model_selection.candidates[sel]],
    ["holdout", "2010 (held out)", facts.holdout[sel]],
    ["out_of_sample", "Full out-of-sample period", facts.out_of_sample[sel]],
  ];
  const cell = (v, ci) => `${fmt.pct1(v)}${ci ? ` <span class="ci">[${fmt.pct1(ci[0])}, ${fmt.pct1(ci[1])}]</span>` : ""}`;
  const signed = (v) => `${v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(1)}`;
  const diff = (d) => `${d.mean >= 0 ? "+" : ""}${signed(d.mean)} pts <span class="ci">[${signed(d.ci[0])}, ${signed(d.ci[1])}]</span>`;
  const rows = W.map(([key, label, nc]) => {
    const b = facts.baselines[key], pd = facts.paired_differences[key]["0.5"];
    return `<tr><td>${label}</td><td class="best">${cell(nc.yes_kept["0.5"], nc.yes_kept_ci && nc.yes_kept_ci["0.5"])}</td>` +
      `<td>${cell(b.client_ranking_logistic.yes_kept["0.5"], b.client_ranking_logistic.yes_kept_ci["0.5"])}</td>` +
      `<td>${cell(b.call_cap.yes_kept["0.5"])}</td>` +
      `<td>${diff(pd.next_call_minus_client_ranking)}</td><td>${diff(pd.next_call_minus_call_cap)}</td></tr>`;
  }).join("");
  document.getElementById("compare-table").innerHTML =
    `<caption class="visually-hidden">Share of subscriptions kept at half the calls</caption>` +
    `<thead><tr><th>Period</th><th>The Last Call</th><th>Client scoring</th><th>Max-N-calls rule</th>` +
    `<th>vs. client scoring</th><th>vs. max-N rule</th></tr></thead><tbody>${rows}</tbody>`;
  const ho = facts.paired_differences.holdout["0.5"].next_call_minus_client_ranking;
  document.getElementById("compare-note").innerHTML =
    `Differences use a paired bootstrap, so the same resampled clients are scored under each policy. ` +
    `In 2010 the next-call model and client scoring are tied (${diff(ho)}). That year half of all contacted clients said yes, ` +
    `leaving little room to rank. The max-N rule is interpolated between whole caps, which a call centre can do by mixing caps at random.`;
}

function fairTable(facts) {
  const rows = Object.entries(facts.example_rule.by_age).map(([g, v]) =>
    `<tr><td>${g}</td><td>${fmt.int(v.clients)}</td><td>${fmt.int(v.subscribed)}</td>` +
    `<td>${fmt.pct1(v.yes_kept_share)}</td><td>${fmt.pct1(v.calls_share)}</td><td>${fmt.pct1(v.contacted_share)}</td></tr>`).join("");
  document.getElementById("fair-table").innerHTML =
    `<caption class="visually-hidden">Effect of the 5% rule by age group</caption>` +
    `<thead><tr><th>Age group</th><th>Clients</th><th>Subscribed</th><th>Subscriptions captured</th>` +
    `<th>Calls still made</th><th>Clients still contacted</th></tr></thead><tbody>${rows}` +
    `<tr><td>All</td><td>${fmt.int(facts.money_rule.totals.clients)}</td><td>${fmt.int(facts.money_rule.totals.yes)}</td>` +
    `<td>${fmt.pct1(facts.example_rule.yes_kept)}</td><td>${fmt.pct1(facts.example_rule.calls_share)}</td><td></td></tr></tbody>`;
}

function guardStrip(facts) {
  const strip = document.getElementById("guard-strip");
  const months = Object.entries(facts.safeguards.months);
  strip.innerHTML = months.map(([p, v]) => {
    const cls = v.status === "active" ? "" : v.status.includes("alarm") ? "shadow alarm" : "shadow";
    const label = `${MONTHS[+p.slice(5, 7) - 1]} ${p.slice(2, 4)}`;
    return `<div class="col"><div class="m ${cls}" tabindex="0" data-p="${p}"></div><div class="lbl">${label}</div></div>`;
  }).join("");
  strip.querySelectorAll(".m").forEach((el) => {
    const v = facts.safeguards.months[el.dataset.p];
    const shift = v.shift === null ? "" : `<br>Last month's odds off by ×${Math.exp(Math.abs(v.shift)).toFixed(2)} ` +
      `(${v.shift > 0 ? "under" : "over"}-predicted)`;
    const html = `<b>${periodLabel(el.dataset.p)}</b><br>${v.status[0].toUpperCase() + v.status.slice(1)}${shift}`;
    el.setAttribute("aria-label", `${periodLabel(el.dataset.p)}: ${v.status}`);
    el.addEventListener("mousemove", (e) => showTip(html, e));
    el.addEventListener("mouseleave", hideTip);
    el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); showTip(html, { clientX: r.right, clientY: r.bottom }); });
    el.addEventListener("blur", hideTip);
  });
  legend(document.getElementById("guard-legend"), [
    { color: "--series-1", label: "Active: rule applied", kind: "chip" },
    { pattern: "repeating-linear-gradient(45deg, var(--shadow-chip) 0 3px, transparent 3px 6px)", label: "Shadow: scored, no calls cut", kind: "chip" },
    { color: "--series-2", label: "Orange outline: drift alarm", kind: "chip" },
  ]);
}

function pilotTable(pilot) {
  const rows = Object.entries(pilot.by_value_to_cost).map(([ratio, v]) =>
    `<tr><td>${ratio}×</td><td>${fmt.pct1(v.b.break_even)}</td><td>${fmt.pct1(v.b.planning_success_rate)}</td>` +
    `<td class="best">${fmt.int(v.b.flagged_calls_to_make)}</td><td>${fmt.pct1(v.b.exact_power)}</td>` +
    `<td>${v.b.months_at_oos_volume.toFixed(1)}</td><td>${fmt.int(v.a.n_per_arm)}</td></tr>`).join("");
  document.getElementById("pilot-table").innerHTML =
    `<caption class="visually-hidden">Pilot sample sizes</caption>` +
    `<thead><tr><th>Deposit value ÷ call cost</th><th>Break-even success rate</th><th>Expected rate of flagged calls</th>` +
    `<th>Flagged calls to make</th><th>Exact power</th><th>Months at 2009–10 volume</th>` +
    `<th>Clients per arm for an A/B test</th></tr></thead><tbody>${rows}</tbody>`;
  document.getElementById("pilot-note").innerHTML =
    `Expected rate = success rate of the calls the rule cut out-of-sample. Flagged calls to make: smallest number with at least 80% ` +
    `exact binomial power for a one-sided 5% test. Months: at the average number of flagged calls per month in 2009–10, half of them made. ` +
    `A/B column: clients per arm to detect the gain in net value per client (two-sided 5%, 80% power). At higher call volumes the pilot is proportionally shorter.`;
}

function verifyGrid(v) {
  const items = [
    ["Status", v.passed ? "All checks passed" : "Checks failing"],
    ["Figures recomputed", `${fmt.int(v.figures)} × ${v.implementations}`],
    ["Model coefficients, Python vs R", `${v.glm_terms} agree`],
    ["Largest coefficient difference", `${v.glm_max_rel_diff.toExponential(0)} relative`],
    ["Pilot sample sizes, Python vs R", `${v.pilot_figures} agree`],
    ["Calendar vs ECB Euribor", `${v.external_mean_abs_diff} pts`],
  ];
  document.getElementById("verify-grid").innerHTML = items.map(([k, val]) => `<div><dt>${k}</dt><dd>${val}</dd></div>`).join("");
}

// ------------------------------------------------------------------ main
async function main() {
  const [facts, curves, monthly, verification, pilot] = await Promise.all(
    ["facts", "curves", "monthly", "verification", "pilot"].map(load));
  bindFacts({ ...facts, verification });
  hero(facts);
  aucChart(facts);
  countingChart(facts);
  driftCharts(monthly);
  const getScenario = simulator(facts, curves);
  compareTable(facts);
  guardStrip(facts);
  pilotTable(pilot);
  fairTable(facts);
  verifyGrid(verification);
  setupAnalyst({ facts, getScenario, fmt });
}

main().catch((err) => {
  console.error(err);
  const box = document.createElement("p");
  box.className = "wrap";
  box.textContent = `Could not load the study data (${err.message}). If you opened index.html from disk, serve the folder instead: python -m http.server`;
  document.getElementById("main").prepend(box);
});
