import argparse
import json
import os
import webbrowser
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd


MODEL_DIR = 'research_paper_outputs_v3'
DEFAULT_DATASET = 'dataset/cpu_data.csv'
LOW_UTILIZATION_THRESHOLD = 0.30
WASTE_THRESHOLD = 0.20
CPU_COST_PER_UNIT_MONTH = 35.0
MEMORY_COST_PER_UNIT_MONTH = 8.0
MIN_HISTORY_DAYS = 7
RECOMMENDATION_VALID_DAYS = 7
IRREGULAR_CV_THRESHOLD = 1.0


def parse_dict_column(value):
    if pd.isna(value) or str(value).strip() in ('', '{}', 'nan'):
        return {}
    try:
        value = str(value).replace('"', "'")
        return json.loads(value.replace("'", '"'))
    except (ValueError, TypeError, json.JSONDecodeError):
        import ast
        try:
            return ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return {}


def parse_distribution(value):
    if pd.isna(value) or str(value).strip() in ('', '[]'):
        return []
    import re
    import ast
    text = re.sub(r'\s+', ',', str(value).strip())
    text = re.sub(r'\[,', '[', text)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [float(item) for item in re.findall(r'[\d.e+\-]+', str(value))]


def load_metrics(filepath, max_rows=200_000, window_days=30):
    print(f'Loading VM history from {filepath}')
    chunk_size = 50_000
    window_start = None
    if window_days:
        time_chunks = pd.read_csv(
            filepath,
            usecols=['start_time'],
            chunksize=chunk_size,
        )
        latest_time = max(
            pd.to_numeric(chunk['start_time'], errors='coerce').max()
            for chunk in time_chunks
        )
        window_start = latest_time - window_days * 24 * 60 * 60 * 1_000_000

    required_columns = [
        'machine_id', 'start_time', 'end_time', 'priority',
        'scheduling_class', 'collection_type', 'resource_request',
        'average_usage', 'maximum_usage', 'cycles_per_instruction',
        'memory_accesses_per_instruction', 'cpu_usage_distribution',
        'tail_cpu_usage_distribution', 'failed',
    ]
    chunks = []
    for chunk in pd.read_csv(
        filepath,
        usecols=required_columns,
        chunksize=chunk_size,
        low_memory=False,
    ):
        chunk['start_time'] = pd.to_numeric(chunk['start_time'], errors='coerce')
        if window_start is not None:
            chunk = chunk[chunk['start_time'] >= window_start]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise ValueError('No records were found in the requested history window.')
    data = pd.concat(chunks, ignore_index=True)
    print(f'Using the latest {window_days} days: {len(data):,} records' if window_days else f'Loaded {len(data):,} records')
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42).reset_index(drop=True)

    for column in ['resource_request', 'average_usage', 'maximum_usage']:
        if column not in data.columns:
            continue
        expanded = pd.DataFrame(
            data[column].map(parse_dict_column).tolist(), index=data.index
        )
        expanded.columns = [f'{column}_{name}' for name in expanded.columns]
        data = pd.concat([data.drop(columns=column), expanded], axis=1)

    for column in ['cpu_usage_distribution', 'tail_cpu_usage_distribution']:
        if column not in data.columns:
            continue
        parsed = data[column].map(parse_distribution)
        data[f'{column}_mean'] = parsed.map(lambda values: np.mean(values) if values else np.nan)
        data[f'{column}_p95'] = parsed.map(lambda values: np.percentile(values, 95) if values else np.nan)
        data.drop(columns=column, inplace=True)

    numeric_columns = [
        'start_time', 'end_time', 'priority', 'scheduling_class',
        'collection_type', 'cycles_per_instruction',
        'memory_accesses_per_instruction', 'failed',
        'resource_request_cpus', 'resource_request_memory',
        'average_usage_cpus', 'average_usage_memory',
        'maximum_usage_cpus', 'maximum_usage_memory',
        'cpu_usage_distribution_mean', 'cpu_usage_distribution_p95',
        'tail_cpu_usage_distribution_mean', 'tail_cpu_usage_distribution_p95',
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')

    data['duration'] = (data['end_time'] - data['start_time']).clip(lower=0)
    data['priority_tier'] = pd.cut(
        data['priority'], bins=[-1, 99, 115, 500], labels=[0, 1, 2]
    ).astype(float)
    data['machine_id'] = data['machine_id'].fillna('unknown').astype(str)
    return data


def build_vm_history(data):
    grouped = data.groupby('machine_id', dropna=False)
    report = grouped.agg(
        observations=('machine_id', 'size'),
        current_cpu=('resource_request_cpus', 'sum'),
        current_memory=('resource_request_memory', 'sum'),
        observed_avg_cpu=('average_usage_cpus', 'mean'),
        observed_peak_cpu=('maximum_usage_cpus', 'max'),
        observed_avg_memory=('average_usage_memory', 'mean'),
        observed_peak_memory=('maximum_usage_memory', 'max'),
        tail_cpu_p95=('tail_cpu_usage_distribution_p95', 'mean'),
        cpu_usage_std=('average_usage_cpus', 'std'),
        memory_usage_std=('average_usage_memory', 'std'),
        first_observation=('start_time', 'min'),
        last_observation=('start_time', 'max'),
        duration=('duration', 'sum'),
        priority=('priority', 'median'),
        scheduling_class=('scheduling_class', 'median'),
        collection_type=('collection_type', 'median'),
        cycles_per_instruction=('cycles_per_instruction', 'median'),
        memory_accesses_per_instruction=('memory_accesses_per_instruction', 'median'),
        failed_jobs=('failed', 'sum'),
    ).reset_index()

    report['priority_tier'] = pd.cut(
        report['priority'], bins=[-1, 99, 115, 500], labels=[0, 1, 2]
    ).astype(float)
    report['cpu_efficiency'] = report['observed_avg_cpu'] / report['current_cpu'].replace(0, np.nan)
    report['mem_efficiency'] = report['observed_avg_memory'] / report['current_memory'].replace(0, np.nan)
    report['tail_cpu_ratio'] = report['tail_cpu_p95'] / report['observed_avg_cpu'].replace(0, np.nan)
    report['cpu_efficiency'] = report['cpu_efficiency'].clip(0, 1).fillna(0)
    report['mem_efficiency'] = report['mem_efficiency'].clip(0, 1).fillna(0)
    report['tail_cpu_ratio'] = report['tail_cpu_ratio'].replace([np.inf, -np.inf], np.nan).clip(0, 10).fillna(0)
    report['failure_rate'] = report['failed_jobs'] / report['observations'].clip(lower=1)
    report['history_days'] = (
        report['last_observation'] - report['first_observation']
    ) / (24 * 60 * 60 * 1_000_000)
    report['cpu_variation'] = report['cpu_usage_std'] / report['observed_avg_cpu'].replace(0, np.nan)
    report['memory_variation'] = report['memory_usage_std'] / report['observed_avg_memory'].replace(0, np.nan)
    report[['cpu_variation', 'memory_variation']] = report[
        ['cpu_variation', 'memory_variation']
    ].replace([np.inf, -np.inf], np.nan).fillna(0)
    report['irregular_workload'] = (
        (report['cpu_variation'] > IRREGULAR_CV_THRESHOLD) |
        (report['memory_variation'] > IRREGULAR_CV_THRESHOLD)
    )
    return report


def score_and_recommend(report):
    cpu_model = joblib.load(os.path.join(MODEL_DIR, 'best_cpu_peak_model.pkl'))
    memory_model = joblib.load(os.path.join(MODEL_DIR, 'best_memory_model.pkl'))
    failure_model = joblib.load(os.path.join(MODEL_DIR, 'best_failure_model.pkl')) if os.path.exists(
        os.path.join(MODEL_DIR, 'best_failure_model.pkl')
    ) else None
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

    feature_names = list(scaler.feature_names_in_)
    features = report.reindex(columns=feature_names, fill_value=0).copy()
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.fillna(0)
    scaled = pd.DataFrame(scaler.transform(features), columns=feature_names)

    report['predicted_peak_cpu'] = np.maximum(0, cpu_model.predict(scaled))
    report['predicted_peak_memory'] = np.maximum(0, memory_model.predict(scaled))
    scaled_values = scaled.to_numpy()
    cpu_tree_predictions = np.asarray([tree.predict(scaled_values) for tree in cpu_model.estimators_])
    memory_tree_predictions = np.asarray([tree.predict(scaled_values) for tree in memory_model.estimators_])
    report['cpu_model_agreement'] = (
        1 - np.std(cpu_tree_predictions, axis=0) /
        np.maximum(np.abs(report['predicted_peak_cpu']), 1e-12)
    ).clip(0, 1)
    report['memory_model_agreement'] = (
        1 - np.std(memory_tree_predictions, axis=0) /
        np.maximum(np.abs(report['predicted_peak_memory']), 1e-12)
    ).clip(0, 1)
    if failure_model is not None:
        report['failure_risk'] = failure_model.predict_proba(scaled)[:, 1]
    else:
        report['failure_risk'] = report['failure_rate'].clip(0, 1)
    report['predicted_cpu_utilization'] = report['predicted_peak_cpu'] / report['current_cpu'].replace(0, np.nan)
    report['predicted_memory_utilization'] = report['predicted_peak_memory'] / report['current_memory'].replace(0, np.nan)
    report[['predicted_cpu_utilization', 'predicted_memory_utilization']] = report[
        ['predicted_cpu_utilization', 'predicted_memory_utilization']
    ].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)

    report['recommended_cpu'] = np.maximum(report['predicted_peak_cpu'] * 1.25, 0.01)
    report['recommended_memory'] = np.maximum(report['predicted_peak_memory'] * 1.25, 0.01)
    report['status'] = np.where(
        (report['predicted_cpu_utilization'] < WASTE_THRESHOLD) &
        (report['predicted_memory_utilization'] < WASTE_THRESHOLD),
        'WASTED',
        np.where(
            (report['predicted_cpu_utilization'] < LOW_UTILIZATION_THRESHOLD) |
            (report['predicted_memory_utilization'] < LOW_UTILIZATION_THRESHOLD),
            'UNDERUTILIZED', 'IN_USE'
        ),
    )
    insufficient = report['history_days'] < MIN_HISTORY_DAYS
    report.loc[insufficient, 'status'] = 'INSUFFICIENT_HISTORY'
    report['alert_engineer'] = report['status'] != 'IN_USE'
    report['risk_level'] = np.where(
        (report['failure_risk'] > 0.10) | report['irregular_workload'],
        'High',
        np.where(report['failure_risk'] > 0.02, 'Medium', 'Low'),
    )
    report.loc[report['irregular_workload'] & ~insufficient, 'status'] = 'IRREGULAR_WORKLOAD'
    report.loc[insufficient, ['recommended_cpu', 'recommended_memory']] = np.nan
    report['estimated_monthly_saving'] = (
        (report['current_cpu'] - report['recommended_cpu']).clip(lower=0) * CPU_COST_PER_UNIT_MONTH +
        (report['current_memory'] - report['recommended_memory']).clip(lower=0) * MEMORY_COST_PER_UNIT_MONTH
    ).round(2)
    report['confidence_score'] = (
        100 * (0.5 * report['cpu_model_agreement'] +
                0.5 * report['memory_model_agreement']) *
        (report['history_days'] / MIN_HISTORY_DAYS).clip(0, 1) *
        (1 - report['failure_risk'].clip(0, 1))
    ).round(1)
    report['confidence_label'] = 'Model agreement, not an accuracy guarantee'
    report['analysis_timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    report['recommendation_expires_utc'] = (
        datetime.now(timezone.utc) + pd.Timedelta(days=RECOMMENDATION_VALID_DAYS)
    ).isoformat()
    report['advisory_only'] = True
    report['manual_review_required'] = (
        report['alert_engineer'] | report['irregular_workload'] |
        (report['confidence_score'] < 70) | insufficient
    )
    return report.sort_values(['alert_engineer', 'estimated_monthly_saving'], ascending=[False, False])


def write_outputs(report):
    output_dir = 'prediction_outputs'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    csv_path = os.path.join(output_dir, f'vm_rightsizing_{timestamp}.csv')
    json_path = os.path.join(output_dir, f'vm_rightsizing_{timestamp}.json')
    html_path = os.path.join(output_dir, f'vm_dashboard_{timestamp}.html')

    report.to_csv(csv_path, index=False)
    report.to_json(json_path, orient='records', indent=2)
    dashboard = report[report['alert_engineer']].copy()
    dashboard.to_html(html_path, index=False, classes='vm-table', table_id='vm-table')
    with open(html_path, 'r', encoding='utf-8') as file:
        table = file.read()
    style = '''<style>body{font-family:Arial;margin:32px;color:#17202a}h1{color:#0b3954}.warning{background:#fff4cc;border:1px solid #e0b400;padding:14px;margin:16px 0}.vm-table{border-collapse:collapse;width:100%}.vm-table th,.vm-table td{border:1px solid #d9e2ec;padding:8px;text-align:right}.vm-table th{background:#0b3954;color:white}.vm-table tr:nth-child(even){background:#f4f7f9}</style>'''
    warning = (
        '<div class="warning"><strong>Advisory only:</strong> this dashboard never changes infrastructure. '
        'Review historical usage and risk before acting. Confidence means model agreement, not an accuracy guarantee. '
        f'Recommendations expire after {RECOMMENDATION_VALID_DAYS} days and require fresh analysis.</div>'
    )
    with open(html_path, 'w', encoding='utf-8') as file:
        file.write(f'<html><head>{style}</head><body><h1>VM Right-Sizing Dashboard</h1>{warning}<p>Flagged VMs: {len(dashboard)} of {len(report)}</p>{table}</body></html>')
    print(f'CSV report: {csv_path}')
    print(f'JSON report: {json_path}')
    print(f'Engineer dashboard: {html_path}')
    webbrowser.open(os.path.abspath(html_path))
    return csv_path, json_path, html_path


def main():
    parser = argparse.ArgumentParser(description='Review historical VM utilization and recommend right-sizing.')
    parser.add_argument('--input', default=DEFAULT_DATASET, help='CSV metrics export')
    parser.add_argument('--max-rows', type=int, default=200_000, help='Rows to sample for low-memory machines')
    parser.add_argument('--window-days', type=int, default=30, help='Recent history window; use 0 for the full dataset')
    args = parser.parse_args()

    report = score_and_recommend(build_vm_history(load_metrics(args.input, args.max_rows, args.window_days)))
    print(f'Flagged {int(report.alert_engineer.sum())} of {len(report)} VMs for engineer review.')
    print(f'Estimated monthly saving across flagged VMs: ${report.loc[report.alert_engineer, "estimated_monthly_saving"].sum():,.2f}')
    write_outputs(report)


if __name__ == '__main__':
    main()