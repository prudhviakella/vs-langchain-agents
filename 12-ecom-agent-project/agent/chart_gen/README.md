# chart_gen — a table to Plotly figures

```
supervisor ──A2A message/stream {question, columns, rows}──► chart_gen (AgentCore runtime ecom_chart_gen)
                                                               ├─ progress: charting
                                                               └─ {chart_type, insight, figures[0..3]}  or {"error"}
   one gpt-6-sol call (structured output) ─► sanitize in CODE ─► figures safe for Plotly.newPlot
```

Ported from ACT's chart_gen: the same judgement (chart or not, how many figures, series vs
figures, dual axis, box for outliers, funnel for drop-off, sunburst for the category tree), with
retail examples. Enforced in code, not asked for: script/handler/URL vectors scrubbed from every
string, unknown trace types dropped, line figures with fewer than 3 points dropped. A chart
failure completes the task with `{"error"}` — the answer goes out without a chart.

## Deploy

```bash
cd agent/chart_gen
python deploy.py
```

## Test

```bash
cd agent/chart_gen
python -m pytest tests -q
```

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `CHART_MODEL` | `gpt-6-sol` | the model |
| `CHART_MAX_FIGURES` | 3 | figures per chart |
| `CHART_MAX_ROWS_TO_MODEL` | 300 | rows the model reads (the cap is stated to it) |
