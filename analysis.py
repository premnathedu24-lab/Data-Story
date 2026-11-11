"""Local descriptive analysis. No network, storage, model calls, or code execution."""
import csv
import html
import io
import math
import re
import statistics
from datetime import datetime, timezone

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10000
MAX_COLUMNS = 30
MAX_CELL = 2000

class DataError(ValueError):
    pass

def read_csv(data: bytes):
    if not data or len(data) > MAX_BYTES:
        raise DataError('Upload a non-empty CSV no larger than 5 MB.')
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise DataError('Please save the CSV with UTF-8 encoding and try again.') from exc
    if '\x00' in text:
        raise DataError('The file contains binary content, not a supported CSV.')
    try:
        sample = text[:65536]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
        except csv.Error:
            first = next((line for line in sample.splitlines() if line.strip()), '')
            counts = {d: first.count(d) for d in (',', ';', '\t', '|')}
            delimiter = max(counts, key=counts.get) if max(counts.values(), default=0) else ','
        reader = csv.reader(io.StringIO(text, newline=''), delimiter=delimiter, strict=True)
        headers = next(reader)
        headers = [h.strip() for h in headers]
        if not headers or len(headers) > MAX_COLUMNS:
            raise DataError('Use a CSV with 1–30 columns and a consistent delimiter.')
        if any(not h or len(h) > 120 for h in headers):
            raise DataError('Use non-empty column names no longer than 120 characters.')
        if len(set(h.casefold() for h in headers)) != len(headers):
            raise DataError('Column names must be unique, including letter case.')
        rows = []
        skipped = 0
        for values in reader:
            if not values:
                skipped += 1
                continue
            if len(values) != len(headers):
                raise DataError('A row has a different number of fields from the header. Check delimiters and quoting.')
            if any(len(v) > MAX_CELL for v in values):
                raise DataError('A cell is too long. Keep each cell under 2,001 characters.')
            if len(rows) >= MAX_ROWS:
                raise DataError('Use at most 10,000 data rows for this demo.')
            rows.append(values)
    except StopIteration as exc:
        raise DataError('The CSV is empty.') from exc
    except csv.Error as exc:
        raise DataError('The CSV could not be parsed. Check quoting and use one consistent delimiter (comma, semicolon, tab or pipe).') from exc
    if not rows:
        raise DataError('The CSV needs at least one data row below its header.')
    return headers, rows, skipped

def number(value):
    value = value.strip().replace(',', '').replace('$', '').replace('₹', '').replace('%', '')
    if not value or value.lower() in ('na','n/a','null','nan','none','inf','-inf','-'):
        return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except ValueError:
        return None

def sensitive_label(name):
    tokens = re.sub(r'([a-z])([A-Z])', r'\1_\2', name).lower()
    return bool(re.search(r'(^|[^a-z])(id|uuid|code|account|email|phone|name|address|postcode|zip|ssn|password|token)([^a-z]|$)', tokens))

def fmt(x):
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return 'not available'
    if x != 0 and (abs(x) >= 1e9 or abs(x) < 0.001):
        return f'{x:.4g}'
    return f'{x:,.2f}'.rstrip('0').rstrip('.')

def profile(data: bytes):
    headers, rows, skipped = read_csv(data)
    n = len(rows)
    duplicates = n - len(set(tuple(row) for row in rows))
    warnings = []
    if skipped:
        warnings.append(f'{skipped} completely blank line(s) were skipped. Nonblank data rows were not removed.')
    if duplicates:
        warnings.append(f'{duplicates} duplicate row(s) after the first occurrence; included in all calculations.')
    if n < 10:
        warnings.append('This is a tiny dataset. Do not treat the findings as a reliable trend or representative sample.')
    columns = []
    for j, name in enumerate(headers):
        raw = [row[j] for row in rows]
        present = [v.strip() for v in raw if v.strip()]
        missing = n - len(present)
        distinct = len(set(present))
        item = dict(name=name, missing=missing, missing_pct=100 * missing / n,
                    distinct=distinct, kind='text / mixed', numeric=None, histogram=None)
        if missing:
            warnings.append(f'{name}: {missing}/{n} values blank ({fmt(100 * missing/n)}%).')
        numbers = [number(v) for v in present]
        valid_numbers = [x for x in numbers if x is not None]
        all_numeric = bool(present) and len(valid_numbers) / len(present) >= 0.95
        identifier = sensitive_label(name)
        # Conservative safeguard: unique integers may be IDs, years, or measurements.
        unique_integer = all_numeric and distinct == len(present) and all(x.is_integer() for x in numbers)
        leading_zero = any(re.match(r'^[+-]?0\d+$', v) for v in present)
        if identifier or (unique_integer and n >= 5) or leading_zero:
            item['kind'] = 'identifier / ambiguous (skipped)'
            warnings.append(f'{name}: automatic numeric analysis skipped because it may be an identifier. This heuristic can also skip genuine measurements.')
        elif not present:
            item['kind'] = 'all blank'
        elif all_numeric:
            # Scale to avoid intermediate summation overflow.
            numbers = valid_numbers
            scale = max(abs(x) for x in numbers)
            mean = (math.fsum(x / scale for x in numbers) / len(numbers) * scale) if scale else 0.0
            median = statistics.median(x / scale for x in numbers) * scale if scale else 0.0
            item['kind'] = 'numeric'
            item['numeric'] = dict(count=len(numbers), mean=mean, median=median,
                                   minimum=min(numbers), maximum=max(numbers))
            lo, hi = min(numbers), max(numbers)
            scaled = [x / scale for x in numbers] if scale else numbers
            low, high = min(scaled), max(scaled)
            if low == high:
                item['histogram'] = [{'range': fmt(lo), 'count': len(numbers)}]
            else:
                bins = min(6, len(numbers))
                width = (high-low) / bins
                counts = [0] * bins
                for x in scaled:
                    counts[min(bins-1, int((x-low)/width))] += 1
                item['histogram'] = [dict(range=f'{fmt((low+i*width)*scale)} to {fmt((low+(i+1)*width)*scale)}', count=c)
                                     for i, c in enumerate(counts)]
        elif distinct <= 10:
            item['kind'] = 'categorical / text'
        if any(v.lower() in ('na', 'n/a', 'null', 'nan', 'none', 'inf', '-inf') for v in present):
            warnings.append(f'{name}: contains missing-value markers or non-finite tokens. Only blank/whitespace cells count as missing; this column is not automatically cleaned.')
        columns.append(item)
    findings = []
    for c in columns:
        if c['numeric'] and len(findings) < 3:
            s = c['numeric']
            findings.append(f"{c['name']}: average {fmt(s['mean'])}, median {fmt(s['median'])}, range {fmt(s['minimum'])}–{fmt(s['maximum'])}, across {s['count']} nonblank values. Units are not inferred.")
    if not findings:
        findings.append('No unambiguous numeric columns were selected. Review the column profile; no numeric story has been invented.')
    return dict(rows=n, column_count=len(headers), duplicate_rows=duplicates,
                missing_cells=sum(c['missing'] for c in columns), columns=columns,
                findings=findings, warnings=warnings,
                method='UTF-8 CSV with automatically detected comma, semicolon, tab or pipe delimiter; blank/whitespace cells treated as missing; exact duplicate rows retained. Numeric metrics use nonblank finite values only when every nonblank value is numeric. IDs and ambiguous numeric columns are conservatively skipped. Text values are not included in the exported report.')

def report_html(p, title='Client data overview', author=''):
    e = lambda v: html.escape(str(v), quote=True)
    lines = []
    for c in p['columns']:
        s = c['numeric']
        lines.append('<tr>' + ''.join(f'<td>{e(v)}</td>' for v in [c['name'], c['kind'],
             f"{c['missing']}/{p['rows']}", fmt(c['missing_pct'])+'%', c['distinct'],
             fmt(s['mean']) if s else '—', fmt(s['median']) if s else '—']) + '</tr>')
    findings = ''.join(f'<li>{e(f)}</li>' for f in p['findings'])
    warnings = ''.join(f'<li>{e(w)}</li>' for w in p['warnings']) or '<li>No blank values or duplicate rows detected. This is not a full data-quality audit.</li>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#172b3a;max-width:1000px;margin:48px auto;padding:0 24px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:32px}}.tag{{color:#08786b;text-transform:uppercase;font-size:12px;letter-spacing:2px}}.meta{{color:#526779}}.stats{{display:flex;gap:18px;flex-wrap:wrap}}.stat{{padding:18px;background:#eff7f5;border-radius:10px;min-width:120px}}.stat b{{display:block;font-size:26px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #dce7e7;text-align:left;padding:10px;overflow-wrap:anywhere}}.table{{overflow-x:auto}}.note{{background:#f5f7fa;padding:18px;border-radius:8px}}@media print{{body{{margin:0;max-width:none}}.table{{overflow:visible}}}}</style></head><body>
<div class="tag">Data Story · descriptive report</div><h1>{e(title)}</h1><p class="meta">{e(author)} · Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
<div class="stats"><div class="stat"><b>{p['rows']:,}</b>data rows</div><div class="stat"><b>{p['column_count']}</b>columns</div><div class="stat"><b>{p['missing_cells']:,}</b>blank cells</div><div class="stat"><b>{p['duplicate_rows']:,}</b>duplicate rows retained</div></div>
<h2>What the data shows</h2><ul>{findings}</ul><h2>Data-quality notes</h2><ul>{warnings}</ul>
<h2>Column profile</h2><div class="table"><table><thead><tr><th>Column</th><th>Detected type</th><th>Blank values</th><th>Blank %</th><th>Distinct nonblank</th><th>Mean</th><th>Median</th></tr></thead><tbody>{''.join(lines)}</tbody></table></div>
<h2>Method and limitations</h2><p>{e(p['method'])}</p><div class="note">Local, rule-based explanation—not an AI-generated report. Descriptive analysis only: no causal claims, forecasts, business recommendations, or statistical significance. Types and units need human review before sharing. Raw source rows are not embedded; retain the original CSV separately for audit. Column names and aggregate metrics may still be confidential. Review this export before sending it to a client.</div></body></html>'''
