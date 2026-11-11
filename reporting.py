"""Standalone HTML/SVG output; no remote chart library or JavaScript required."""
import copy
import html
import math
import pandas as pd
from analysis import report_html as base_report, fmt

def esc(value):
    return html.escape(str(value), quote=True)

def status_text(guidance):
    return {
        'exploration': 'Exploratory ideas only; no user goal has been inferred.',
        'advisory': 'Conditional model candidates; no models have been trained or compared.',
        'needs_target': 'Choose a meaningful target column for more specific guidance.',
        'invalid_target': 'The selected target is not suitable as currently profiled; review the reasons below.',
        'needs_review': 'Context or data review is needed before treating these suggestions as an actionable modelling plan.',
        'insufficient_data': 'The available observations do not support the proposed validation step.',
        'unsupported_goal': 'The selected goal is not supported.'
    }.get(guidance.get('status'), str(guidance.get('status','Review needed')))

def svg_chart(chart, include_labels=False):
    points = chart.get('points', [])
    if not points:
        return ''
    labels = [p['label'] if include_labels or not chart.get('category_labels') else f'Category {i+1}' for i,p in enumerate(points)]
    values = [p['value'] for p in points]
    title = esc(chart['title'])
    svg = []
    palette = ['#087f8c','#bd5b2b','#6e60aa','#4b7c49','#af5472','#737884','#ae8424','#559c9f']
    if chart['kind']=='pie':
        total = math.fsum(values)
        angle=-math.pi/2
        for i,v in enumerate(values):
            nxt=angle+v/total*2*math.pi
            x1,y1=125+95*math.cos(angle),125+95*math.sin(angle)
            x2,y2=125+95*math.cos(nxt),125+95*math.sin(nxt)
            large=1 if nxt-angle>math.pi else 0
            svg.append(f'<path d="M125 125 L{x1:.3f} {y1:.3f} A95 95 0 {large} 1 {x2:.3f} {y2:.3f} Z" fill="{palette[i%len(palette)]}"><title>{esc(labels[i])}: {fmt(v)} ({fmt(100*v/total)}%)</title></path>')
            angle=nxt
        legend=''.join(f'<li><span style="color:{palette[i%len(palette)]}">●</span> {esc(label)}: {fmt(v)} ({fmt(100*v/total)}%)</li>' for i,(label,v) in enumerate(zip(labels,values)))
        picture=f'<svg viewBox="0 0 250 250" style="max-width:280px" role="img" aria-label="{title}">{"".join(svg)}</svg><ul>{legend}</ul>'
    elif chart['kind']=='line':
        # x positions use actual elapsed time, not row index.
        from datetime import datetime
        ds=[datetime.fromisoformat(p['label']) for p in points]
        span=(ds[-1]-ds[0]).total_seconds()
        scale=max(abs(v) for v in values) or 1.0
        normalized=[v/scale for v in values]
        lo,hi=min(normalized),max(normalized)
        def ypos(v): return 105 if hi==lo else 180-(v-lo)/(hi-lo)*150
        coords=[(55+(d-ds[0]).total_seconds()/span*600,ypos(v)) for d,v in zip(ds,normalized)]
        svg.append('<line x1="55" x2="655" y1="180" y2="180" stroke="#9aaeb5"/>')
        svg.append('<polyline fill="none" stroke="#087f8c" stroke-width="3" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in coords)+'"/>')
        for i,(x,y) in enumerate(coords):
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#087f8c"><title>{esc(labels[i])}: {fmt(values[i])}</title></circle>')
        svg.append(f'<text x="5" y="30" font-size="12">{esc(fmt(max(values)))}</text><text x="5" y="180" font-size="12">{esc(fmt(min(values)))}</text>')
        svg.append(f'<text x="55" y="205" font-size="11">{esc(labels[0][:10])}</text><text x="570" y="205" font-size="11">{esc(labels[-1][:10])}</text>')
        picture=f'<svg viewBox="0 0 720 220" role="img" aria-label="{title}">{"".join(svg)}</svg>'
    else:
        peak=max(values) or 1
        height=36*len(values)+25
        for i,(label,v) in enumerate(zip(labels,values)):
            y=15+i*36
            display=label[:30]+('…' if len(label)>30 else '')
            svg.append(f'<text x="5" y="{y+16}" font-size="12">{esc(display)}</text><rect x="240" y="{y}" width="{v/peak*390:.2f}" height="24" fill="#087f8c" rx="3"><title>{esc(label)}: {fmt(v)}</title></rect><text x="{250+v/peak*390:.2f}" y="{y+16}" font-size="12">{esc(fmt(v))}</text>')
        picture=f'<svg viewBox="0 0 720 {height}" role="img" aria-label="{title}">{"".join(svg)}</svg>'
    category_chart = chart.get('category_labels', False)
    total = math.fsum(values) if category_chart else 0
    extra_header = '<th>% of nonblank rows</th>' if category_chart else ''
    value_rows = []
    for label,v in zip(labels,values):
        extra = f'<td>{esc(fmt(100*v/total))}%</td>' if category_chart and total else ''
        value_rows.append(f'<tr><td>{esc(label)}</td><td>{esc(fmt(v))}</td>{extra}</tr>')
    table='<details><summary>Chart values</summary><table><tr><th>Category, range or time</th><th>Value</th>'+extra_header+'</tr>'+''.join(value_rows)+'</table></details>'
    return f'<section style="break-inside:avoid;margin:28px 0"><h3>{title}</h3>{picture}<p>{esc(chart["explanation"])}</p>{table}</section>'

def report_html(p,title='Client data overview',author='',include_labels=False,training_result=None):
    clean=copy.deepcopy(p)
    for c in clean['columns']:
        c['kind']=c.get('semantic_kind',c['kind'])
    clean['method']=clean['method'].replace('Text values are not included in the exported report.', 'Raw rows are not embedded. Actual category labels are included only if the user explicitly enabled them; otherwise chart labels are generic.')
    base=base_report(clean,title,author)
    sections=['<h2>Categorical data and encoding suggestions</h2><p>Suggestions only; no encoder has been fitted or applied. Categories are distinct values, not necessarily distinct storage data types.</p>']
    for c in p['columns']:
        sections.append(f'<h3>{esc(c["name"])} · {esc(c.get("semantic_kind",c["kind"]))}</h3><p>{esc(c.get("encoding","Review column meaning first."))}</p>')
        if c.get('representation'):
            sections.append(f'<p>{esc(c["representation"])}</p>')
    sections.append(f'<h2>Bias, privacy and fitness-for-use review</h2><p><strong>{esc(p.get("review_status","Context required"))}</strong></p><p>{esc(p.get("review_note",""))}</p><p>Column-name heuristics can miss sensitive content. An upload passing validation is not a guarantee that data is safe, lawful, representative, or suitable for a model.</p>')
    if p.get('data_review'):
        sections.append('<ul>'+''.join(f'<li>{esc(note)}</li>' for note in p['data_review'])+'</ul>')
    sections.append('<h2>Charts and what their values mean</h2>')
    if not include_labels:
        sections.append('<p>Category labels are generic in this export by default; counts and percentages remain visible. This reduces accidental disclosure but does not anonymize the report. Column names, dates and aggregates may still be sensitive.</p>')
    else:
        sections.append('<p>Actual category labels were explicitly enabled for this export. Review them for confidential or identifying information before sharing.</p>')
    sections.extend(svg_chart(ch,include_labels) for ch in p.get('charts',[]))
    if not p.get('time_charts'):
        sections.append('<p>No time-series chart was selected. A line chart needs a recognised date/time column and a suitable numeric measurement; row order alone is not treated as time.</p>')
    if training_result:
        metrics = training_result.get('results', {})
        metric_rows = ''.join('<tr><th>'+esc(name)+'</th>'+''.join('<td>'+esc(fmt(value))+'</td>' for value in values.values())+'</tr>' for name,values in metrics.items())
        metric_headers = list(next(iter(metrics.values())).keys()) if metrics else []
        sections.append('<h2>Machine-learning experiment results</h2>')
        sections.append('<p>Task: '+esc(training_result.get('task'))+' · Target: '+esc(training_result.get('target'))+' · Best model on this evaluation split: <strong>'+esc(training_result.get('best_model'))+'</strong>.</p>')
        if metric_rows:
            sections.append('<div class="table"><table><thead><tr><th>Model</th>'+''.join('<th>'+esc(h)+'</th>' for h in metric_headers)+'</tr></thead><tbody>'+metric_rows+'</tbody></table></div>')
        if training_result.get('excluded_features'):
            rows = ''.join('<tr><td>'+esc(item.get('feature',''))+'</td><td>'+esc(item.get('reason',''))+'</td></tr>' for item in training_result['excluded_features'])
            sections.append('<h3>Selected columns excluded from modeling</h3><div class="table"><table><tr><th>Column</th><th>Reason</th></tr>'+rows+'</table></div>')
        if training_result.get('classification_report'):
            sections.append('<h3>Per-class classification report</h3><div class="table">'+pd.DataFrame(training_result['classification_report']).T.to_html(escape=True)+'</div>')
        if training_result.get('target_warnings'):
            sections.append('<h3>Target review warnings</h3><ul>'+''.join('<li>'+esc(w)+'</li>' for w in training_result['target_warnings'])+'</ul>')
        if training_result.get('task') == 'forecasting':
            sections.append('<p>Forecast frequency: '+esc(training_result.get('frequency','not available'))+' · Duplicate timestamps aggregated: '+esc(training_result.get('duplicate_timestamps_aggregated',0))+'</p>')
        if training_result.get('task') == 'clustering':
            sections.append('<p>Clustering is exploratory and unsupervised. Silhouette scores are internal separation measures, not prediction accuracy or evidence of valid business segments.</p>')
        preds=training_result.get('predictions')
        if preds is not None:
            sections.append('<h3>Held-out prediction examples</h3><div class="table">'+preds.head(50).to_html(index=False,escape=True)+'</div>')
        future=training_result.get('future_forecast')
        if future is not None:
            sections.append('<h3>Forecast output</h3><div class="table">'+future.to_html(index=False,escape=True)+'</div>')
        sections.append('<p>Metrics are specific to this split and dataset; they do not guarantee future performance. Review leakage, representativeness, validation design and domain suitability.</p>')
    g=p.get('model_guidance')
    if g:
        sections.append(f'<h2>Possible ML project and model candidates</h2><p>Goal: {esc(g.get("task","Explore"))}. Target: {esc(p.get("selected_target") or "not selected")}. {esc(status_text(g))}</p><p>No models have been trained or compared. These are starting candidates, not a best-model verdict.</p>')
        for key,label in [('possible_projects','Possible projects'),('candidates','Candidate baselines/models'),('rationale','Why consider them'),('evaluation','How to evaluate'),('cautions','Cautions')]:
            values=g.get(key,[])
            if values:
                sections.append(f'<h3>{esc(label)}</h3><ul>'+''.join(f'<li>{esc(v)}</li>' for v in values)+'</ul>')
    return base.replace('<h2>Method and limitations</h2>',''.join(sections)+'<h2>Method and limitations</h2>')
