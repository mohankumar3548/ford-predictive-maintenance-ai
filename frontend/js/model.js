/* Ford Line Health Monitor: sample data generator and algorithm for Ford Motor Company. */
const TOOL = {
  num: 9, area: "Operations", product: "Predictive Maintenance", seed: 99,
  title: "Ford Line Health Monitor", company: "Ford Motor Company",
  why: "A Ford assembly plant depends on presses, welding robots and conveyors, and one unplanned breakdown can stop the whole line. This tool reads machine sensors, predicts which machine fails next and books the repair into planned downtime.",
  statement: "Unexpected equipment failures are causing production downtime and increasing maintenance costs. Develop an AI system that analyzes machine sensor data, maintenance history and production schedules to identify failure patterns and recommend preventive maintenance actions.",
  dataNote: "24 machines on one body-and-assembly line, 120 days of daily sensor readings and 3 years of failure history (sample data).",
  steps: [
    ["Collect sensor and maintenance data", "Vibration, temperature, motor current, work orders, schedule"],
    ["Build health features", "Each sensor vs the machine's own healthy baseline (z-score)"],
    ["Anomaly detection", "Health index from combined z-scores, flag above 3"],
    ["Remaining useful life", "Trend of the health index to the failure level"],
    ["7-day failure probability", "Logistic model trained on past failure windows"],
    ["Failure patterns", "K-means on past failure signatures"],
    ["Schedule maintenance", "Book each job into the last planned window before failure"],
    ["Work orders", "AI analyst drafts technician instructions"]
  ],
  params: [
    { id: "dt", label: "Unplanned downtime cost", min: 20, max: 300, step: 10, value: 120, fmt: v => "$" + v + "K / hr" },
    { id: "buf", label: "Safety buffer before predicted failure", min: 0, max: 10, step: 1, value: 3, fmt: v => v + " days" }
  ],
  questions: ["Which machine should we fix first?", "What is causing most failures on this line?", "Write a work order for the top-risk machine", "How much money does this plan save?"],
  generate(r) {
    const types = [["Stamping press", 4], ["Welding robot", 8], ["Paint robot", 3], ["Conveyor drive", 5], ["CNC machining", 4]];
    const machines = []; types.forEach(([t, n]) => { for (let i = 0; i < n; i++) machines.push({ id: `${t.split(" ")[0].slice(0, 4).toUpperCase()}-${String(i + 1).padStart(2, "0")}`, type: t }); });
    const modes = { bearing: [1, 0.25, 0.1], overheat: [0.2, 1, 0.3], electrical: [0.1, 0.3, 1] };
    const degr = [[2, "bearing", 78, 0.17], [9, "overheat", 92, 0.12], [14, "bearing", 100, 0.1], [19, "electrical", 100, 0.17], [6, "electrical", 110, 0.1], [22, "overheat", 70, 0.05]];
    machines.forEach((m, i) => { const d = degr.find(x => x[0] === i); m.mode = d ? d[1] : null; m.start = d ? d[2] : 999; m.rate = d ? d[3] : 0; m.lastPM = r.int(10, 160); m.baseV = r.u(2, 4); m.baseT = r.u(55, 70); m.baseC = r.u(30, 60);
      m.days = S.range(120).map(t => { const k = Math.max(0, t - m.start) * m.rate * 1.15, w = m.mode ? modes[m.mode] : [0, 0, 0]; return { v: m.baseV * (1 + r.n(0, 0.06) + 0.09 * k * w[0]), t: m.baseT * (1 + r.n(0, 0.03) + 0.032 * k * w[1]), c: m.baseC * (1 + r.n(0, 0.05) + 0.07 * k * w[2]) }; }); });
    // historical windows: z features + days since PM, label fail within 7 days
    const hist = [], fails = [];
    for (let i = 0; i < 2600; i++) { const mode = r.wpick(Object.keys(modes), [5, 3, 2]); const w = modes[mode]; const sev = r.chance(0.18) ? r.u(4, 10) : r.u(0, 3.2); const pm = r.int(5, 220); const zv = r.n(sev * w[0], 0.8), zt = r.n(sev * w[1], 0.8), zc = r.n(sev * w[2], 0.8); const p = S.sigmoid(-5 + 0.55 * (zv + zt + zc) + 0.008 * pm); const f = r() < p ? 1 : 0; hist.push({ x: [zv, zt, zc, pm], f }); if (f && Math.max(zv, zt, zc) > 2.5) fails.push({ z: [zv, zt, zc], mode }); }
    return { machines, hist, fails };
  },
  analyze(D, P, r) {
    const ms = D.machines, now = 119, failHI = 8;
    const model = M.logit(D.hist.map(h => h.x), D.hist.map(h => h.f), { iters: 300, lr: 0.5 });
    const auc = M.auc(D.hist.map(h => model.predict(h.x)), D.hist.map(h => h.f));
    const rows = ms.map(m => {
      const base = m.days.slice(0, 30); const st = k => { const v = base.map(d => d[k]); return [S.mean(v), S.sd(v)]; };
      const [mv, sv] = st("v"), [mt, stt] = st("t"), [mc, sc] = st("c");
      const z = m.days.map(d => [(d.v - mv) / sv, (d.t - mt) / stt, (d.c - mc) / sc]);
      const hi = z.map(q => Math.max(0, (q[0] + q[1] + q[2]) / 1.4));
      const sm = hi.map((_, i) => S.mean(hi.slice(Math.max(0, i - 4), i + 1)));
      const last = z.slice(-5).reduce((a, q) => a.map((v, j) => v + q[j] / 5), [0, 0, 0]);
      const recent = sm.slice(-21), fit = M.ols(recent.map((_, i) => [i]), recent); const slope = fit.beta[1];
      const cur = sm[now]; const rul = slope > 0.02 ? Math.max(0, (failHI - cur) / slope) : Infinity;
      const p7 = model.predict([...last, m.lastPM + 0]);
      const dom = last.indexOf(Math.max(...last)); const sig = cur > 2 ? ["Vibration rising: bearing or alignment wear", "Temperature rising: cooling or lubrication fault", "Current rising: motor or electrical fault"][dom] : "Normal";
      return { m, hi: sm, cur, slope, rul, p7, sig, dom, last };
    }).sort((a, b) => b.p7 - a.p7);
    // failure pattern clustering
    const km = M.kmeans(D.fails.map(f => f.z), 3, RNG(5)); const pat = S.range(3).map(c => { const g = D.fails.filter((_, i) => km.labels[i] === c); const ctr = km.centers[c]; const d = ctr.indexOf(Math.max(...ctr)); return { n: g.length, name: ["Bearing and alignment wear", "Overheating", "Electrical and motor faults"][d], ctr }; }).sort((a, b) => b.n - a.n);
    // schedule: planned windows every Sunday (day index 6,13,...) 8h, plus shift gaps on Wednesdays 4h
    const windows = []; for (let d = now + 1; d <= now + 60; d++) { const dow = d % 7; if (dow === 6) windows.push({ d, hrs: 8, kind: "Sunday shutdown" }); if (dow === 2) windows.push({ d, hrs: 4, kind: "Midweek gap" }); }
    const repairHrs = { bearing: 6, overheat: 4, electrical: 3, null: 2 };
    const jobs = rows.filter(x => x.p7 > 0.25 || x.rul < 45).map(x => { const due = now + (isFinite(x.rul) ? x.rul : 60) - P.buf; const need = [6, 4, 3][x.dom]; const w = [...windows].reverse().find(w => w.d <= due && w.hrs >= need) || null; return { x, due, need, w, stopNow: !w }; });
    const prevented = jobs.length, unplannedHrs = S.sum(jobs.map(j => j.need * 3 + 6)), plannedHrs = S.sum(jobs.filter(j => !j.stopNow).map(() => 0)) + S.sum(jobs.filter(j => j.stopNow).map(j => j.need));
    const saved = (unplannedHrs - plannedHrs) * P.dt * 1000;
    const tone = p => p > 0.6 ? "bad" : p > 0.3 ? "warn" : "";
    const top3 = rows.slice(0, 3);
    return {
      steps: [`${ms.length} machines × 120 days, ${D.hist.length} historical windows`, "3 sensor z-scores per machine-day", `${rows.filter(x => x.cur > 3).length} machines with health index above 3`, `Shortest RUL: ${rows.filter(x => isFinite(x.rul)).sort((a, b) => a.rul - b.rul)[0]?.m.id} at ${rows.filter(x => isFinite(x.rul)).sort((a, b) => a.rul - b.rul)[0]?.rul.toFixed(0)} days`, `Model AUC ${auc.toFixed(3)}; top risk ${rows[0].m.id} at ${F.pct(rows[0].p7, 0)}`, pat.map(p => `${p.name} ${F.pct(p.n / D.fails.length, 0)}`).join(", "), `${jobs.length} jobs booked, ${jobs.filter(j => j.stopNow).length} needing an immediate stop`, "Ready"],
      kpis: [
        { label: "Machines at risk", value: String(jobs.length), note: `of ${ms.length} on the line`, tone: jobs.length ? "bad" : "good" },
        { label: "Highest 7-day risk", value: F.pct(rows[0].p7, 0), note: `${rows[0].m.id} · ${rows[0].m.type}`, tone: "bad" },
        { label: "Downtime avoided", value: F.num(unplannedHrs - plannedHrs) + " hrs", note: "Unplanned stops turned into planned work", tone: "good" },
        { label: "Saving", value: F.money(saved), note: `At $${P.dt}K per downtime hour`, tone: "good" }
      ],
      sections: [
        { title: "Health index, three riskiest machines", note: "Combined sensor drift from each machine's own healthy baseline. The red line is the level where past machines failed.", chart: C.line(S.range(120).map(i => "D" + (i + 1)), top3.map((x, i) => ({ values: x.hi, tone: ["c3", "c1", "c4"][i] })).concat([{ values: new Array(120).fill(null), ref: failHI }]), { fmt: v => v.toFixed(0), every: 15, yMin: 0 }) + C.legend(top3.map((x, i) => ({ label: `${x.m.id} (${x.m.type})`, tone: ["c3", "c1", "c4"][i] }))) },
        { pair: [
          { title: "7-day failure probability", chart: C.bar(rows.slice(0, 10).map(x => ({ label: `${x.m.id} · ${x.m.type}`, value: x.p7, tone: tone(x.p7) })), { fmt: v => F.pct(v, 0), max: 1, labelW: 180 }) },
          { title: "How machines have failed", note: `${D.fails.length} past failures grouped by their sensor signature (K-means).`, chart: C.bar(pat.map(p => ({ label: p.name, value: p.n / D.fails.length, tone: "c3" })), { fmt: v => F.pct(v, 0), labelW: 190 }) }
        ] },
        { title: "Maintenance schedule", note: "Each job goes into the last planned window that still leaves the safety buffer before predicted failure.", html: T.table(["Machine", "Type", "7-day risk", "Days to failure", "Signal", "Repair hrs", "Book into"], jobs.map(j => [j.x.m.id, j.x.m.type, T.pill(F.pct(j.x.p7, 0), tone(j.x.p7) || "mut"), isFinite(j.x.rul) ? j.x.rul.toFixed(0) : "Stable", j.x.sig, j.need, j.stopNow ? T.pill("Stop now", "bad") : `Day ${j.w.d - now} · ${j.w.kind}`]), { right: [3, 5] }) }
      ],
      recs: jobs.slice(0, 4).map(j => ({ title: `${j.x.m.id}: ${j.x.sig.split(":")[1]?.trim() || "inspect"}`, body: `${j.x.m.type} at ${F.pct(j.x.p7, 0)} risk of failing within 7 days${isFinite(j.x.rul) ? `, about ${j.x.rul.toFixed(0)} days of life left` : ""}. ${j.stopNow ? "No planned window comes early enough, so stop it at the next shift change." : `Book the ${j.need}-hour job into the ${j.w.kind.toLowerCase()} in ${j.w.d - now} days.`}`, impact: j.stopNow ? "Stop now" : `Day ${j.w.d - now}`, tone: j.stopNow ? "bad" : "warn" })).concat([{ title: `Target ${pat[0].name.toLowerCase()} across the line`, body: `${F.pct(pat[0].n / D.fails.length, 0)} of past failures follow this pattern. Add vibration checks and lubrication to every Sunday shutdown for the machine types most exposed.`, impact: "Prevent repeats", tone: "" }]),
      context: { model_auc: +auc.toFixed(3), downtime_cost_per_hour: P.dt * 1000, buffer_days: P.buf, machines: rows.slice(0, 10).map(x => ({ machine: x.m.id, type: x.m.type, failure_prob_7d: +x.p7.toFixed(2), health_index: +x.cur.toFixed(1), days_to_failure: isFinite(x.rul) ? Math.round(x.rul) : "stable", signal: x.sig, days_since_maintenance: x.m.lastPM })), failure_patterns: pat.map(p => ({ pattern: p.name, share: +(p.n / D.fails.length).toFixed(2) })), jobs: jobs.map(j => ({ machine: j.x.m.id, repair_hours: j.need, booked: j.stopNow ? "stop now" : `day ${j.w.d - now} ${j.w.kind}` })), downtime_hours_avoided: Math.round(unplannedHrs - plannedHrs), saving: Math.round(saved) }
    };
  }
};
