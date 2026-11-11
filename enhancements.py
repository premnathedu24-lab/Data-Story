"""Descriptive, local extensions. No training, automatic recoding, or fairness certification."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
import re
from analysis import profile as base_profile, read_csv, number, fmt, sensitive_label

REVIEW_NOTE = ('Representation counts are descriptive, not a fairness test. Uneven counts do not prove discriminatory bias; equal counts do not prove safety. Fairness needs the intended use, collection process, relevant groups and outcomes, and—when a model exists—appropriate per-group error evaluation. Privacy, consent, provenance and leakage require separate review.')


def parse_dates(values):
    if not values:
        return None
    out = []
    try:
        for value in values:
            if not re.match(r'^\d{4}-\d{2}-\d{2}(?:$|[T ])', value):
                return None
            out.append(datetime.fromisoformat(value.replace('Z', '+00:00')))
        aware = [d.tzinfo is not None for d in out]
        if any(aware) and not all(aware):
            return None
        return [d.astimezone(timezone.utc) for d in out] if all(aware) else out
    except (ValueError, OverflowError):
        return None


def encoding_for(column):
    kind = column['semantic_kind']
    count = column['category_count']
    common = 'Fit preprocessing on training data only, keep the same mapping on validation/test data, and explicitly handle missing and unseen values. These are suggestions; no data has been encoded.'
    if column['kind'].startswith('identifier'):
        return 'Review as a possible identifier. Usually exclude row IDs from model features; do not infer an ordinal meaning from numeric codes. ' + common
    if kind == 'all blank':
        return 'All values are blank. Investigate collection or consider removing the unusable feature; do not invent categories.'
    if column['date_like']:
        return 'Recognised ISO-style date/time values. Confirm timezone, frequency and meaning; use justified calendar/lag features for the intended task, not arbitrary category codes. ' + common
    if kind == 'constant':
        return 'One observed value: this feature has no variation in this sample. Consider dropping it, unless the domain explains why it should remain. ' + common
    if kind.startswith('binary'):
        return 'Two-category feature: a documented 0/1 mapping or a one-hot encoding is suitable. Neither code implies rank. For a classification target, encode labels separately and never include the target among input features. ' + common
    if kind == 'mixed numeric/text':
        return 'Mixed numeric and non-numeric tokens were detected. Resolve whether they are labels, formatting, missing-value markers or entry errors before choosing an encoding. Do not silently coerce them.'
    if kind == 'numeric':
        return 'No categorical encoding is needed if these are genuine measurements. Scaling may help distance-based and linear models; confirm meaning, units and whether numbers are actually codes. ' + common
    if count > 20 or kind == 'high-cardinality / free text':
        return 'Many distinct or long text values: inspect for IDs or free text before encoding. Avoid blindly creating thousands of one-hot columns. For true categories, consider train-fitted rare-category grouping or hashing; text may need a separate text representation. ' + common
    return 'Unordered categories: one-hot encoding is a reasonable starting point for input features. Use ordinal encoding only when a genuine order is explicitly confirmed; do not impose an arbitrary ranking. ' + common


def enrich_profile(data):
    p = base_profile(data)
    headers, rows, _ = read_csv(data)
    charts = []
    parsed_dates = {}
    for j, c in enumerate(p['columns']):
        present = [row[j].strip() for row in rows if row[j].strip()]
        counts = Counter(present)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        dates = parse_dates(present)
        if dates:
            parsed_dates[j] = dates
        nums = [number(v) for v in present]
        valid_nums = [v for v in nums if v is not None]
        numeric = bool(present) and len(valid_nums) / len(present) >= 0.95
        c.update(is_numeric_values=numeric, category_count=len(counts),
                 minority_count=min(counts.values()) if counts else None,
                 date_like=dates is not None, category_counts=[], representation='',
                 semantic_kind=c['kind'], sensitive=sensitive_label(c['name']))
        if not present:
            semantic = 'all blank'
        elif c['kind'].startswith('identifier'):
            semantic = c['kind']
        elif dates:
            semantic = 'date/time hint'
        elif len(counts) == 1:
            semantic = 'constant'
        elif numeric and set(nums).issubset({0.0, 1.0}):
            semantic = 'binary numeric codes (confirm meaning)'
        elif numeric:
            semantic = 'numeric'
        elif any(v is not None for v in nums) and not numeric:
            semantic = 'mixed numeric/text'
        elif len(counts) == 2:
            semantic = 'binary categorical'
        elif len(counts) <= 20 and max(map(len, present)) <= 80:
            semantic = 'nominal categorical (order not inferred)'
        else:
            semantic = 'high-cardinality / free text'
        c['semantic_kind'] = semantic
        if semantic in ('nominal categorical (order not inferred)', 'binary categorical'):
            c['kind'] = 'categorical / text'
        categorical = ('categorical' in semantic or semantic.startswith('binary') or semantic == 'constant')
        if categorical and not c['sensitive']:
            c['category_counts'] = [dict(value=value, count=count, percent=100*count/len(present)) for value,count in ordered[:8]]
            dominant = ordered[0][1]/len(present)
            smallest = ordered[-1][1]
            c['representation'] = f"{len(counts)} observed categories across {len(present)} nonblank rows; largest share {fmt(100*dominant)}%; smallest category {smallest} row(s). Percentages exclude {c['missing']} blank row(s)."
            if len(counts) > 1 and dominant >= 0.8:
                c['representation'] += ' Dominance review flag: one category has at least 80% of nonblank rows. This is a screening heuristic, not proof of bias.'
            if len(counts) > 1 and smallest < 5:
                c['representation'] += ' Some categories have fewer than five observations; estimates or class-based validation may be unstable.'
            if len(counts) <= 20 and len([ch for ch in charts if ch['kind']=='bar']) < 2:
                points = [dict(label=v, value=count) for v,count in ordered[:7]]
                rest = sum(count for _,count in ordered[7:])
                if rest:
                    points.append(dict(label='[Other categories combined]', value=rest))
                explanation = f"Counts of {c['name']} among {len(present)} nonblank rows. Largest category: {ordered[0][1]} rows ({fmt(100*dominant)}%); {c['missing']} blank row(s) excluded. Bars count rows, not importance or causal effects."
                charts.append(dict(kind='bar', title=f"{c['name']}: category counts", points=points, explanation=explanation, category_labels=True, column=c['name']))
                if 2 <= len(counts) <= 6 and not any(ch['kind']=='pie' for ch in charts):
                    charts.append(dict(kind='pie', title=f"{c['name']}: share of nonblank rows", points=points,
                                       explanation=f"Each slice is its category's count divided by {len(present)} nonblank rows. Slices sum to 100%; blank rows are excluded. This is representation in this file, not a fairness or safety verdict.", category_labels=True, column=c['name']))
        c['encoding'] = encoding_for(c)
        if c['histogram'] and semantic in ('numeric', 'constant') and not c['sensitive'] and len([ch for ch in charts if ch['kind']=='histogram']) < 2:
            charts.append(dict(kind='histogram',title=f"{c['name']}: numeric distribution", points=[dict(label=b['range'],value=b['count']) for b in c['histogram']],
                               explanation=f"Each bar counts nonblank observations in a numeric interval. Values are rounded for display; the calculation uses original numeric values. {c['missing']} blank row(s) excluded. Distribution shape alone does not establish normality or statistical significance.",category_labels=False,column=c['name']))
    numeric_cols = [(j,c) for j,c in enumerate(p['columns']) if c['numeric'] and c['semantic_kind']=='numeric' and not c['sensitive']]
    p['time_charts'] = []
    if parsed_dates and numeric_cols:
        date_j = next(iter(parsed_dates))
        metric_j, metric = numeric_cols[0]
        groups = defaultdict(list)
        for row in rows:
            if not row[date_j].strip():
                continue
            dt = parse_dates([row[date_j].strip()])
            value = number(row[metric_j])
            if dt and value is not None:
                groups[dt[0]].append(value)
        if len(groups) >= 2:
            def average(vs):
                scale=max(abs(v) for v in vs)
                return math.fsum(v/scale for v in vs)/len(vs)*scale if scale else 0.0
            points = [dict(label=dt.isoformat(),value=average(vs)) for dt,vs in sorted(groups.items())]
            # Keep report size bounded; do not silently sample a long series.
            if len(points) <= 100:
                desc = f"Mean {metric['name']} for each exact observed {headers[date_j]} timestamp, sorted chronologically; duplicate timestamps are averaged, not summed. {sum(len(vs) for vs in groups.values())} rows with valid date/metric pairs contribute. First mean {fmt(points[0]['value'])}; last mean {fmt(points[-1]['value'])}. No missing dates are filled. Dates are ISO-style parsing hints, not confirmation of a forecasting task or regular frequency."
                chart = dict(kind='line',title=f"{metric['name']} by {headers[date_j]}",points=points,explanation=desc,category_labels=False,column=metric['name'])
                charts.append(chart)
                p['time_charts'].append(chart)
            else:
                p['warnings'].append('Time chart skipped: more than 100 distinct timestamps. Choose an explicit aggregation period in a later version; no silent downsampling was applied.')
    p['charts'] = charts
    p['review_note'] = REVIEW_NOTE
    named = [c['name'] for c in p['columns'] if c['sensitive']]
    p['data_review'] = [
        'Sampling and collection context was not supplied: representativeness relative to a real population has not been assessed.',
        'Protected groups, outcomes, intended decisions and potential harms must be defined before a fairness assessment. No fairness audit or safe-use certification has run.',
        'Resolve missing values, duplicate/entity structure and possible answer leakage before training. Fit preprocessing on training folds, not on the full uploaded file.'
    ]
    if named:
        p['data_review'].append('Possible identifier or sensitive column-name matches: ' + ', '.join(named) + '. Names are only a heuristic; other sensitive content may be missed. Review before sharing or modelling.')
    p['review_status'] = 'Context required — fairness, privacy and fitness for use are not certified.'
    p['encoding_notes'] = 'Suggestions only. Feature encoders and target-label encoders serve different purposes. Do not transform data or create an ordered code merely because values have names.'
    p['method'] += ' Extended profile: categorical counts trim surrounding whitespace but do not merge case or spelling variants. Category percentages exclude blank rows. An 80% dominant share flags review only, not unfair bias. No encoding or model training is performed.'
    return p
