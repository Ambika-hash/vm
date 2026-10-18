# """
# ML-Based Predictive Analysis for Resource Utilization in Computing Systems
# ==========================================================================
# VERSION 3 — LEAKAGE-FREE PIPELINE (hidden text bug fixed)

# FIXES FROM VERSION 2:
#   - Removed applymap(clean_text) on entire dataframe — was corrupting numeric
#     columns and causing matplotlib to embed invisible/white text in PDFs
#   - Clean text only applied to object (string) columns only
#   - Removed index ASCII cleaning which caused hidden character artifacts
#   - PDF font embedding forced to Type 42 to prevent hidden glyph substitution
#   - Figure backgrounds explicitly set to white to prevent transparent-over-white
#     text rendering issues in Word/PDF viewers
#   - All axes backgrounds explicitly white
#   - Tight layout replaced with constrained_layout for cleaner PDF output

# PRACTICAL FEATURE SET (no data leakage):
#   - resource_request_cpus      : known at submission
#   - resource_request_memory    : known at submission
#   - priority                   : known at submission
#   - scheduling_class           : known at submission
#   - average_usage_cpus         : mean CPU — NOT the target
#   - average_usage_memory       : mean memory
#   - cpu_efficiency             : avg_cpu / request_cpu — no max involved
#   - mem_efficiency             : avg_mem / request_mem — no max involved
#   - tail_cpu_ratio             : tail_mean / avg_cpu — no max involved
#   - duration                   : how long job ran
#   - cpu_dist_mean/std          : distribution stats — no max involved
#   - priority_tier              : binned priority

# TARGETS:
#   T1: average_usage_cpus   — predict average CPU
#   T2: maximum_usage_cpus   — predict peak CPU (NOT in features)
#   T3: average_usage_memory — predict memory
#   T4: failed               — predict failure (classification)

# HOW TO RUN:
#   pip install pandas numpy scikit-learn matplotlib seaborn joblib
#   Change CSV_PATH at the bottom to your file
#   python ml_resource_prediction_v3.py

# RAM GUIDE:
#   4GB  RAM → MAX_ROWS = 100_000
#   8GB  RAM → MAX_ROWS = 200_000
#   16GB RAM → MAX_ROWS = 400_000
# """

# import pandas as pd
# import numpy as np
# import ast, re, os, warnings
# warnings.filterwarnings('ignore')

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# # ── Force clean PDF font embedding — prevents hidden glyph artifacts ──────
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype']  = 42
# plt.rcParams['svg.fonttype'] = 'none'

# import matplotlib.gridspec as gridspec
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (mean_absolute_error, mean_squared_error,
#                              r2_score, classification_report, roc_auc_score)
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
#                               RandomForestClassifier, GradientBoostingClassifier)
# from sklearn.tree import DecisionTreeRegressor
# import joblib

# # ── Global style — all backgrounds explicitly white ───────────────────────
# plt.rcParams.update({
#     'font.family':       'DejaVu Sans',
#     'axes.titlesize':    13,
#     'axes.labelsize':    11,
#     'xtick.labelsize':   9,
#     'ytick.labelsize':   9,
#     'figure.dpi':        150,
#     'axes.grid':         True,
#     'grid.alpha':        0.3,
#     'figure.facecolor': 'white',
#     'axes.facecolor':   'white',
#     'savefig.facecolor':'white',
#     'savefig.edgecolor':'white',
#     'figure.facecolor':  'white',   # no transparent backgrounds
#     'axes.facecolor':    'white',
#     'savefig.facecolor': 'white',
#     'text.color':        'black',   # force all text black — prevents white text
#     'axes.labelcolor':   'black',
#     'xtick.color':       'black',
#     'ytick.color':       'black',
# })

# PALETTE = ['#2563EB','#16A34A','#DC2626','#D97706','#7C3AED','#0891B2']

# # ══════════════════════════════════════════════════════════════════════════
# # SETTINGS
# # ══════════════════════════════════════════════════════════════════════════
# MAX_ROWS  = 200_000
# N_TREES   = 50
# MAX_DEPTH = 10

# REGRESSORS = {
#     'Linear Regression': LinearRegression(),
#     'Ridge Regression':  Ridge(alpha=1.0),
#     'Decision Tree':     DecisionTreeRegressor(max_depth=6, random_state=42),
#     'Random Forest':     RandomForestRegressor(
#                              n_estimators=N_TREES, max_depth=MAX_DEPTH,
#                              random_state=42, n_jobs=-1),
#     'Gradient Boosting': GradientBoostingRegressor(
#                              n_estimators=N_TREES, max_depth=4,
#                              learning_rate=0.1, random_state=42),
# }
# CLASSIFIERS = {
#     'Random Forest':     RandomForestClassifier(
#                              n_estimators=N_TREES, max_depth=MAX_DEPTH,
#                              random_state=42, n_jobs=-1),
#     'Gradient Boosting': GradientBoostingClassifier(
#                              n_estimators=N_TREES, max_depth=4,
#                              learning_rate=0.1, random_state=42),
# }

# # ══════════════════════════════════════════════════════════════════════════
# # 1. DATA LOADING
# # ══════════════════════════════════════════════════════════════════════════

# def parse_dict_column(val):
#     if pd.isna(val) or str(val).strip() in ('', '{}', 'nan'):
#         return {}
#     try:
#         return ast.literal_eval(str(val).replace('"', "'"))
#     except:
#         try:
#             cpus = re.search(r"'cpus':\s*([\d.e+\-]+)", str(val))
#             mem  = re.search(r"'memory':\s*([\d.e+\-]+)", str(val))
#             return {
#                 'cpus':   float(cpus.group(1)) if cpus else np.nan,
#                 'memory': float(mem.group(1))  if mem  else np.nan
#             }
#         except:
#             return {}

# def parse_distribution(val):
#     if pd.isna(val) or str(val).strip() in ('', '[]'):
#         return []
#     try:
#         cleaned = re.sub(r'\s+', ',', str(val).strip())
#         cleaned = re.sub(r'\[,', '[', cleaned)
#         return ast.literal_eval(cleaned)
#     except:
#         nums = re.findall(r'[\d.e+\-]+', str(val))
#         return [float(x) for x in nums]

# def clean_string_columns(df):
#     """
#     Clean non-ASCII characters from STRING columns ONLY.
#     Do NOT touch numeric columns — that was the bug in v2 causing
#     hidden/corrupted text to appear in generated PDF figures.
#     """
#     for col in df.select_dtypes(include='object').columns:
#         df[col] = df[col].apply(
#             lambda x: x.encode('ascii', 'ignore').decode('ascii')
#             if isinstance(x, str) else x
#         )
#     return df

# def load_and_parse(filepath):
#     print("\n" + "="*60)
#     print("STEP 1: Loading and parsing data")
#     print("="*60)
#     print(f"  File: {filepath}")

#     df = pd.read_csv(filepath, low_memory=False)
#     print(f"  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

#     if len(df) > MAX_ROWS:
#         print(f"  Large dataset — sampling {MAX_ROWS:,} rows (i3 safe mode)")
#         df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
#     else:
#         print(f"  Using all {len(df):,} rows")

#     # Parse dictionary-format columns
#     for col in ['resource_request', 'average_usage', 'maximum_usage',
#                 'random_sample_usage']:
#         if col in df.columns:
#             expanded = df[col].apply(parse_dict_column).apply(pd.Series)
#             expanded.columns = [f"{col}_{k}" for k in expanded.columns]
#             df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

#     # Parse distribution columns — summarise as mean/std only (no max)
#     for col in ['cpu_usage_distribution', 'tail_cpu_usage_distribution']:
#         if col in df.columns:
#             parsed = df[col].apply(parse_distribution)
#             df[f'{col}_mean'] = parsed.apply(lambda x: np.mean(x) if x else np.nan)
#             df[f'{col}_std']  = parsed.apply(lambda x: np.std(x)  if x else np.nan)
#             # NOTE: _max column is computed here but NEVER added to FEATURE_COLS
#             # It is intentionally discarded to maintain leakage-free design
#             df.drop(columns=[col], inplace=True)

#     if 'failed' in df.columns:
#         df['failed'] = (pd.to_numeric(df['failed'], errors='coerce')
#                           .fillna(0).astype(int))

#     # Convert remaining object columns to numeric where possible
#     for col in df.select_dtypes(include='object').columns:
#         df[col] = pd.to_numeric(df[col], errors='ignore')

#     # Clean string columns only — NOT numeric columns
#     df = clean_string_columns(df)

#     # Clean column names — strip whitespace and non-ASCII
#     df.columns = (df.columns
#                     .str.strip()
#                     .str.encode('ascii', 'ignore')
#                     .str.decode('ascii'))

#     print(f"  Parsed shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
#     return df

# # ══════════════════════════════════════════════════════════════════════════
# # 2. FEATURE ENGINEERING — LEAKAGE-FREE
# # ══════════════════════════════════════════════════════════════════════════

# def engineer_features(df):
#     print("\n" + "="*60)
#     print("STEP 2: Engineering features (leakage-free)")
#     print("="*60)

#     if 'start_time' in df.columns and 'end_time' in df.columns:
#         df['duration'] = (
#             pd.to_numeric(df['end_time'],   errors='coerce') -
#             pd.to_numeric(df['start_time'], errors='coerce')
#         ).clip(lower=0)
#         print("  + duration")

#     # cpu_efficiency — uses AVERAGE cpu only, no maximum
#     if ('tail_cpu_usage_distribution_mean' in df.columns and
#             'average_usage_cpus' in df.columns):
#         df['tail_cpu_ratio'] = (
#             df['tail_cpu_usage_distribution_mean'] /
#             df['average_usage_cpus'].replace(0, np.nan)
#         ).clip(0, 10)
#         print("  + tail_cpu_ratio  [tail/avg — no max]")

#     if 'priority' in df.columns:
#         df['priority_tier'] = pd.cut(
#             pd.to_numeric(df['priority'], errors='coerce'),
#             bins=[-1, 99, 115, 500], labels=[0, 1, 2]
#         ).astype(float)
#         print("  + priority_tier")

#     if 'scheduling_class' in df.columns:
#         df['scheduling_class'] = pd.to_numeric(
#             df['scheduling_class'], errors='coerce')

#     print()
#     print("  EXCLUDED (contain maximum_usage_cpus — circular with T2 target):")
#     print("    x cpu_burst_ratio = max / avg")
#     print("    x cpu_range       = max - avg")
#     print(f"\n  Total columns: {df.shape[1]}")
#     return df

# # ══════════════════════════════════════════════════════════════════════════
# # 3. PREPROCESSING
# # ══════════════════════════════════════════════════════════════════════════

# FEATURE_COLS = [
#     'resource_request_cpus',
#     'resource_request_memory',
#     'priority',
#     'scheduling_class',
#     'collection_type',
#     'average_usage_cpus',
#     'average_usage_memory',
#     'cpu_usage_distribution_mean',
#     'cpu_usage_distribution_std',
#     'tail_cpu_usage_distribution_mean',
#     'duration',
#     'cpu_efficiency',
#     'mem_efficiency',
#     'tail_cpu_ratio',
#     'priority_tier',
#     'cycles_per_instruction',
#     'memory_accesses_per_instruction',
#     # maximum_usage_cpus is intentionally NOT here — it is a target
# ]

# def preprocess(df):
#     print("\n" + "="*60)
#     print("STEP 3: Preprocessing")
#     print("="*60)

#     available = [c for c in FEATURE_COLS if c in df.columns]
#     print(f"  Features available: {len(available)}")
#     for f in available:
#         print(f"    - {f}")

#     # Hard leakage guard — abort if target column sneaks into features
#     if 'maximum_usage_cpus' in available:
#         raise ValueError(
#             "LEAKAGE DETECTED: maximum_usage_cpus found in feature set. "
#             "Remove it from FEATURE_COLS immediately."
#         )
#     print("\n  Leakage check PASSED — maximum_usage_cpus not in features")

#     X = df[available].copy()
#     X.fillna(X.median(numeric_only=True), inplace=True)

#     scaler   = StandardScaler()
#     X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#     splits = {}
#     targets = [
#         ('average_usage_cpus',   'cpu_avg',  'regression'),
#         ('maximum_usage_cpus',   'cpu_peak', 'regression'),
#         ('average_usage_memory', 'memory',   'regression'),
#         ('failed',               'failure',  'classification'),
#     ]

#     for target, label, task in targets:
#         if target not in df.columns:
#             print(f"  Skipping {label} — column not found")
#             continue

#         y = df[target].copy()

#         if task == 'classification':
#             y = y.fillna(0).astype(int)
#             if y.nunique() < 2:
#                 continue
#             splits[label] = train_test_split(
#                 X_scaled, y, test_size=0.2, random_state=42, stratify=y)
#         else:
#             y    = pd.to_numeric(y, errors='coerce')
#             mask = y.notna() & (y >= 0)
#             if mask.sum() < 10:
#                 continue
#             splits[label] = train_test_split(
#                 X_scaled[mask], y[mask], test_size=0.2, random_state=42)

#         n = mask.sum() if task == 'regression' else len(y)
#         print(f"  Ready: {label} ({n:,} samples)")

#     return splits, scaler, available

# # ══════════════════════════════════════════════════════════════════════════
# # 4. TRAINING
# # ══════════════════════════════════════════════════════════════════════════

# def train_regressors(X_train, X_test, y_train, y_test, label):
#     print(f"\n  [{label}]")
#     records, trained = [], {}
#     for name, model in REGRESSORS.items():
#         try:
#             print(f"    {name}...", end='', flush=True)
#             model.fit(X_train, y_train)
#             pred = model.predict(X_test)
#             mae  = mean_absolute_error(y_test, pred)
#             rmse = np.sqrt(mean_squared_error(y_test, pred))
#             r2   = r2_score(y_test, pred)
#             records.append({'Model': name, 'MAE': round(mae, 6),
#                             'RMSE': round(rmse, 6), 'R2': round(r2, 4)})
#             trained[name] = (model, pred)
#             print(f" R2={r2:.4f}")
#         except Exception as e:
#             print(f" FAILED: {e}")
#     return (pd.DataFrame(records).sort_values('R2', ascending=False),
#             trained)

# def train_classifiers(X_train, X_test, y_train, y_test):
#     print(f"\n  [failure]")
#     records, trained = [], {}
#     for name, model in CLASSIFIERS.items():
#         try:
#             print(f"    {name}...", end='', flush=True)
#             model.fit(X_train, y_train)
#             pred  = model.predict(X_test)
#             proba = model.predict_proba(X_test)[:, 1]
#             auc   = (roc_auc_score(y_test, proba)
#                      if len(np.unique(y_test)) > 1 else 0.5)
#             rep   = classification_report(y_test, pred, output_dict=True)
#             records.append({
#                 'Model':    name,
#                 'Accuracy': round(rep['accuracy'], 4),
#                 'F1':       round(rep.get('weighted avg', {})
#                                      .get('f1-score', 0), 4),
#                 'ROC_AUC':  round(auc, 4),
#             })
#             trained[name] = (model, pred)
#             print(f" AUC={auc:.4f}")
#         except Exception as e:
#             print(f" FAILED: {e}")
#     return (pd.DataFrame(records).sort_values('ROC_AUC', ascending=False),
#             trained)

# # ══════════════════════════════════════════════════════════════════════════
# # 5. FIGURES
# # All figures: explicit white background, black text — no hidden text
# # ══════════════════════════════════════════════════════════════════════════

# def _save(fig, filename):
#     """Save figure with guaranteed white background and no hidden text."""
#     fig.savefig(
#         filename,
#         bbox_inches='tight',
#         facecolor='white',        # white background always
#         edgecolor='none',
#         dpi=150,
#     )
#     plt.close(fig)
#     print(f"  Saved: {filename}")

# def fig_line(y_test, pred, label, filename):
#     fig, axes = plt.subplots(2, 1, figsize=(14, 8),
#                              facecolor='white',
#                              constrained_layout=True)
#     n         = min(300, len(y_test))
#     actual    = np.array(y_test)[:n]
#     predicted = np.array(pred)[:n]
#     idx       = np.arange(n)

#     ax = axes[0]
#     ax.set_facecolor('white')
#     ax.plot(idx, actual,    color='#2563EB', lw=1.2, label='Actual',    alpha=0.85)
#     ax.plot(idx, predicted, color='#F97316', lw=1.5, label='Predicted', alpha=0.9)
#     ax.set_title(f'Actual vs Predicted — {label}', fontweight='bold', color='black')
#     ax.set_ylabel('Usage Fraction', color='black')
#     ax.legend()

#     ax = axes[1]
#     ax.set_facecolor('white')
#     error = np.abs(actual - predicted)
#     ax.fill_between(idx, 0, error, color='#DC2626', alpha=0.4,
#                     label='Absolute Error')
#     ax.axhline(error.mean(), color='#DC2626', lw=1.5, linestyle='--',
#                label=f'Mean Error = {error.mean():.5f}')
#     ax.set_title('Absolute Prediction Error', color='black')
#     ax.set_ylabel('|Actual - Predicted|', color='black')
#     ax.legend()

#     _save(fig, filename)

# def fig_bar(df_m, title, metrics, filename):
#     n   = len(metrics)
#     fig, axes = plt.subplots(1, n, figsize=(5 * n, 5),
#                              facecolor='white',
#                              constrained_layout=True)
#     if n == 1:
#         axes = [axes]

#     for ax, metric in zip(axes, metrics):
#         if metric not in df_m.columns:
#             continue
#         ax.set_facecolor('white')
#         vals = df_m[metric]
#         bars = ax.barh(df_m['Model'], vals,
#                        color=PALETTE[:len(df_m)],
#                        edgecolor='white', height=0.6)
#         ax.set_xlabel(metric, color='black')
#         ax.set_title(metric, color='black')
#         ax.invert_yaxis()
#         for bar, val in zip(bars, vals):
#             ax.text(bar.get_width() + 0.0005,
#                     bar.get_y() + bar.get_height() / 2,
#                     f'{val:.4f}', va='center', fontsize=8, color='black')

#     fig.suptitle(title, fontsize=13, fontweight='bold', color='black')
#     _save(fig, filename)

# def fig_importance(model, features, label, filename):
#         if not hasattr(model, 'feature_importances_'): return

#         DISPLAY = {
#             'average_usage_cpus':               'Avg CPU',
#             'tail_cpu_usage_distribution_mean': 'Tail Dist',
#             'resource_request_cpus':            'CPU Req',
#             'resource_request_memory':          'Mem Req',
#             'average_usage_memory':             'Avg Mem',
#             'cpu_efficiency':                   'CPU Eff',
#             'mem_efficiency':                   'Mem Eff',
#             'tail_cpu_ratio':                   'Tail Ratio',
#             'cpu_burst_ratio':                  'Burst Ratio',
#             'cpu_range':                        'CPU Range',
#             'duration':                         'Job Dur',
#             'collection_type':                  'Coll Type',
#             'scheduling_class':                 'Sched Cls',
#             'priority_tier':                    'Priority',
#             'cpu_usage_distribution_mean':      'Dist Mean',
#             'cpu_usage_distribution_std':       'Dist Std',
#             'cycles_per_instruction':           'CPI',
#             'memory_accesses_per_instruction':  'MAI',
#         }

#         imp = pd.Series(model.feature_importances_, index=features).nlargest(15).sort_values()
#         imp.index = [DISPLAY.get(f, f[:8]) for f in imp.index]

#         colors = ['#DC2626' if any(k in f for k in ['Burst','Range','Tail'])
#                 else '#2563EB' for f in imp.index]

#         fig, ax = plt.subplots(figsize=(8,6))
#         imp.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
#         ax.set_title('Predictor Weights\n(red = spike features)', fontweight='bold')
#         ax.set_xlabel('Relative Importance')

#         fig.patch.set_facecolor('white')
#         ax.set_facecolor('white')

#         plt.tight_layout()
#         plt.savefig(filename, bbox_inches='tight', dpi=300,
#                     facecolor='white', edgecolor='white', pad_inches=0.1)
#         plt.close()
#         print(f"  Saved: {filename}")

# def fig_eda(df, filename):
#     cols = [c for c in [
#         'resource_request_cpus', 'resource_request_memory',
#         'average_usage_cpus',    'average_usage_memory',
#         'maximum_usage_cpus',    'cpu_efficiency',
#     ] if c in df.columns]
#     if not cols:
#         return

#     fig, axes = plt.subplots(2, 3, figsize=(15, 8),
#                              facecolor='white',
#                              constrained_layout=True)
#     axes = axes.flatten()

#     for i, col in enumerate(cols[:6]):
#         data = df[col].dropna()
#         data = data[np.isfinite(data)]
#         if len(data) == 0:
#             continue
#         ax = axes[i]
#         ax.set_facecolor('white')
#         ax.hist(data, bins=50, color=PALETTE[i % len(PALETTE)],
#                 edgecolor='white', alpha=0.85)
#         DISPLAY = {
#             'resource_request_cpus':  'CPU Request',
#             'resource_request_memory':'Mem Request',
#             'average_usage_cpus':     'Avg CPU',
#             'average_usage_memory':   'Avg Memory',
#             'maximum_usage_cpus':     'Peak CPU',
#             'cpu_efficiency':         'CPU Eff.',
#         }
#         ax.set_title(DISPLAY.get(col, col[:10]), color='black')
#         ax.set_ylabel('Count', color='black')

#     for j in range(len(cols), 6):
#         axes[j].set_visible(False)

#     fig.suptitle('Exploratory Data Analysis',
#                  fontsize=14, fontweight='bold', color='black')
#     _save(fig, filename)

# def fig_correlation(df, filename):
#     key = [c for c in [
#         'resource_request_cpus',  'resource_request_memory',
#         'average_usage_cpus',     'average_usage_memory',
#         'maximum_usage_cpus',
#         'cpu_efficiency',         'mem_efficiency',
#         'tail_cpu_ratio',         'duration',
#         'failed',
#     ] if c in df.columns]

#     if len(key) < 3:
#         return

#     fig, ax = plt.subplots(figsize=(11, 9),
#                            facecolor='white',
#                            constrained_layout=True)
#     ax.set_facecolor('white')
#     sns.heatmap(
#         df[key].corr(),
#         annot=True, fmt='.2f', cmap='coolwarm',
#         center=0, square=True, linewidths=0.5,
#         ax=ax, annot_kws={'size': 8, 'color': 'black'},
#     )
#     ax.set_title(
#         'Feature Correlation Matrix\n'
#         '(maximum_usage_cpus shown for reference only — not used as feature)',
#         fontsize=11, fontweight='bold', color='black')
#     _save(fig, filename)

# def fig_dashboard(results_dict, filename):
#     fig = plt.figure(figsize=(16, 10), facecolor='white')
#     gs  = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.4)

#     configs = [
#         ('cpu_avg',  'R2',      'CPU Average — R\u00b2',   PALETTE[0], gs[0, 0]),
#         ('cpu_peak', 'R2',      'CPU Peak — R\u00b2',      PALETTE[1], gs[0, 1]),
#         ('memory',   'R2',      'Memory — R\u00b2',        PALETTE[2], gs[0, 2]),
#         ('cpu_avg',  'RMSE',    'CPU Average — RMSE',      PALETTE[3], gs[1, 0]),
#         ('cpu_peak', 'RMSE',    'CPU Peak — RMSE',         PALETTE[4], gs[1, 1]),
#         ('failure',  'ROC_AUC', 'Failure — ROC-AUC',       PALETTE[5], gs[1, 2]),
#     ]

#     for key, metric, title, color, pos in configs:
#         ax = fig.add_subplot(pos)
#         ax.set_facecolor('white')
#         df = results_dict.get(key)

#         if df is None or df.empty or metric not in df.columns:
#             ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
#                     transform=ax.transAxes, color='black')
#             ax.set_title(title, fontsize=10, color='black')
#             continue

#         top  = df.nlargest(4, metric)
#         bars = ax.barh(top['Model'], top[metric],
#                        color=color, edgecolor='white')
#         ax.set_title(title, fontsize=10, fontweight='bold', color='black')
#         ax.invert_yaxis()

#         # Explicit black labels on bars — prevents invisible text
#         for bar, val in zip(bars, top[metric]):
#             ax.text(bar.get_width() * 0.5,
#                     bar.get_y() + bar.get_height() / 2,
#                     f'{val:.4f}', va='center', ha='center',
#                     fontsize=7, color='white', fontweight='bold')

#     fig.suptitle('Model Evaluation Dashboard — Leakage-Free Pipeline',
#                  fontsize=14, fontweight='bold', color='black')
#     _save(fig, filename)

# # ══════════════════════════════════════════════════════════════════════════
# # 6. MAIN PIPELINE
# # ══════════════════════════════════════════════════════════════════════════

# def run_pipeline(filepath, output_dir='research_paper_outputs_v3'):
#     os.makedirs(output_dir, exist_ok=True)

#     print("\n" + "█"*60)
#     print("  ML RESOURCE UTILIZATION PIPELINE v3 — LEAKAGE FREE")
#     print("  Hidden text bug fixed — all figures clean")
#     print("█"*60)

#     df = load_and_parse(filepath)
#     df = engineer_features(df)

#     print("\n" + "="*60)
#     print("STEP 4: EDA figures")
#     print("="*60)
#     fig_eda(df,         f'{output_dir}/fig1_distributions.png')
#     fig_correlation(df, f'{output_dir}/fig2_correlation.png')

#     splits, scaler, features = preprocess(df)
#     all_results = {}

#     print("\n" + "="*60)
#     print("STEP 5: Training models")
#     print("="*60)

#     for label in ['cpu_avg', 'cpu_peak', 'memory']:
#         if label not in splits:
#             continue
#         Xtr, Xte, ytr, yte = splits[label]

#         metrics, trained = train_regressors(Xtr, Xte, ytr, yte, label)
#         all_results[label] = metrics

#         best_name          = metrics.iloc[0]['Model']
#         best_model, best_pred = trained[best_name]

#         fig_line(
#             yte, best_pred,
#             f'{label} ({best_name})',
#             f'{output_dir}/fig_{label}_actual_vs_predicted.png'
#         )
#         fig_bar(
#             metrics,
#             f'{label} Prediction',
#             ['MAE', 'RMSE', 'R2'],
#             f'{output_dir}/fig_{label}_comparison.png'
#         )
#         fig_importance(
#             best_model, features, label,
#             f'{output_dir}/fig_{label}_importance.png'
#         )
#         joblib.dump(best_model, f'{output_dir}/best_{label}_model.pkl')

#     if 'failure' in splits:
#         Xtr, Xte, ytr, yte = splits['failure']
#         metrics, _          = train_classifiers(Xtr, Xte, ytr, yte)
#         all_results['failure'] = metrics
#         fig_bar(
#             metrics,
#             'Failure Prediction',
#             ['Accuracy', 'F1', 'ROC_AUC'],
#             f'{output_dir}/fig_failure_comparison.png'
#         )

#     fig_dashboard(all_results, f'{output_dir}/fig_dashboard.png')
#     joblib.dump(scaler, f'{output_dir}/scaler.pkl')

#     print("\n" + "█"*60)
#     print("  PIPELINE COMPLETE")
#     print("█"*60)

#     for label, df_m in all_results.items():
#         print(f"\n  [{label.upper()}]")
#         print(df_m.to_string(index=False))

#     return all_results

# # ══════════════════════════════════════════════════════════════════════════
# # ENTRY POINT
# # ══════════════════════════════════════════════════════════════════════════
# if __name__ == '__main__':
#     CSV_PATH   = "dataset/cpu_data.csv"   # <- CHANGE THIS TO YOUR FILE
#     OUTPUT_DIR = "research_paper_outputs_v3"
#     run_pipeline(CSV_PATH, OUTPUT_DIR)

#----------------------------------------------------------------------------------------------------------------
"""
ML-Based Predictive Analysis for Resource Utilization in Computing Systems
==========================================================================
VERSION 3 — LEAKAGE-FREE PIPELINE (hidden text bug fixed)

FIXES FROM VERSION 2:
  - Removed applymap(clean_text) on entire dataframe — was corrupting numeric
    columns and causing matplotlib to embed invisible/white text in PDFs
  - Clean text only applied to object (string) columns only
  - Removed index ASCII cleaning which caused hidden character artifacts
  - PDF font embedding forced to Type 42 to prevent hidden glyph substitution
  - Figure backgrounds explicitly set to white to prevent transparent-over-white
    text rendering issues in Word/PDF viewers
  - All axes backgrounds explicitly white
  - Tight layout replaced with constrained_layout for cleaner PDF output
  - All raw column names removed from figure axes and titles
  - fig_correlation axes renamed to short display labels
  - fig_eda titles use short display names

PRACTICAL FEATURE SET (no data leakage):
  - resource_request_cpus      : known at submission
  - resource_request_memory    : known at submission
  - priority                   : known at submission
  - scheduling_class           : known at submission
    - collection_type            : workload collection type
  - average_usage_cpus         : mean CPU — NOT the target
  - average_usage_memory       : mean memory
  - cpu_efficiency             : avg_cpu / request_cpu — no max involved
  - mem_efficiency             : avg_mem / request_mem — no max involved
  - tail_cpu_ratio             : tail_mean / avg_cpu — no max involved
  - duration                   : how long job ran
  - cpu_dist_mean/std          : distribution stats — no max involved
  - priority_tier              : binned priority

TARGETS:
  T1: average_usage_cpus   — predict average CPU
  T2: maximum_usage_cpus   — predict peak CPU (NOT in features)
  T3: average_usage_memory — predict memory
  T4: failed               — predict failure (classification)

HOW TO RUN:
  pip install pandas numpy scikit-learn matplotlib seaborn joblib
  Change CSV_PATH at the bottom to your file
  python ml_resource_prediction_v3.py

RAM GUIDE:
  4GB  RAM -> MAX_ROWS = 100_000
  8GB  RAM -> MAX_ROWS = 200_000
  16GB RAM -> MAX_ROWS = 400_000
"""

import pandas as pd
import numpy as np
import ast, re, os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Force clean PDF font embedding — prevents hidden glyph artifacts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42
plt.rcParams['svg.fonttype'] = 'none'

import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, classification_report, roc_auc_score)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeRegressor
import joblib

# Global style — all backgrounds explicitly white
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.titlesize':     13,
    'axes.labelsize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'figure.dpi':         150,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
    'savefig.edgecolor':  'white',
    'text.color':         'black',
    'axes.labelcolor':    'black',
    'xtick.color':        'black',
    'ytick.color':        'black',
})

PALETTE = ['#2563EB', '#16A34A', '#DC2626', '#D97706', '#7C3AED', '#0891B2']

# Global display name mapping — NO raw column names shown in any figure
DISPLAY = {
    'resource_request_cpus':            'CPU Request',
    'resource_request_memory':          'Mem Request',
    'average_usage_cpus':               'Avg CPU',
    'average_usage_memory':             'Avg Memory',
    'maximum_usage_cpus':               'Peak CPU',
    'maximum_usage_memory':             'Peak Memory',
    'cpu_efficiency':                   'CPU Eff.',
    'mem_efficiency':                   'Mem Eff.',
    'tail_cpu_ratio':                   'Tail Ratio',
    'tail_cpu_usage_distribution_mean': 'Tail Dist',
    'cpu_usage_distribution_mean':      'Dist Mean',
    'cpu_usage_distribution_std':       'Dist Std',
    'cpu_burst_ratio':                  'Burst Ratio',
    'cpu_range':                        'CPU Range',
    'duration':                         'Job Duration',
    'collection_type':                  'Coll. Type',
    'scheduling_class':                 'Sched. Class',
    'priority_tier':                    'Priority Tier',
    'priority':                         'Priority',
    'cycles_per_instruction':           'CPI',
    'memory_accesses_per_instruction':  'MAI',
    'failed':                           'Failed',
}

# ===========================================================================
# SETTINGS
# ===========================================================================
MAX_ROWS  = 200_000
N_TREES   = 50
MAX_DEPTH = 10

REGRESSORS = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Decision Tree':     DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest':     RandomForestRegressor(
                             n_estimators=N_TREES, max_depth=MAX_DEPTH,
                             random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(
                             n_estimators=N_TREES, max_depth=4,
                             learning_rate=0.1, random_state=42),
}
CLASSIFIERS = {
    'Random Forest':     RandomForestClassifier(
                             n_estimators=N_TREES, max_depth=MAX_DEPTH,
                             random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
                             n_estimators=N_TREES, max_depth=4,
                             learning_rate=0.1, random_state=42),
}

# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def parse_dict_column(val):
    if pd.isna(val) or str(val).strip() in ('', '{}', 'nan'):
        return {}
    try:
        return ast.literal_eval(str(val).replace('"', "'"))
    except:
        try:
            cpus = re.search(r"'cpus':\s*([\d.e+\-]+)", str(val))
            mem  = re.search(r"'memory':\s*([\d.e+\-]+)", str(val))
            return {
                'cpus':   float(cpus.group(1)) if cpus else np.nan,
                'memory': float(mem.group(1))  if mem  else np.nan
            }
        except:
            return {}

def parse_distribution(val):
    if pd.isna(val) or str(val).strip() in ('', '[]'):
        return []
    try:
        cleaned = re.sub(r'\s+', ',', str(val).strip())
        cleaned = re.sub(r'\[,', '[', cleaned)
        return ast.literal_eval(cleaned)
    except:
        nums = re.findall(r'[\d.e+\-]+', str(val))
        return [float(x) for x in nums]

def clean_string_columns(df):
    """
    Clean non-ASCII characters from STRING columns ONLY.
    Do NOT touch numeric columns.
    """
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(
            lambda x: x.encode('ascii', 'ignore').decode('ascii')
            if isinstance(x, str) else x
        )
    return df

def load_and_parse(filepath):
    print("\n" + "="*60)
    print("STEP 1: Loading and parsing data")
    print("="*60)
    print(f"  File: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    if len(df) > MAX_ROWS:
        print(f"  Large dataset — sampling {MAX_ROWS:,} rows (i3 safe mode)")
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
    else:
        print(f"  Using all {len(df):,} rows")

    for col in ['resource_request', 'average_usage', 'maximum_usage',
                'random_sample_usage']:
        if col in df.columns:
            parsed = df[col].map(parse_dict_column).tolist()
            expanded = pd.DataFrame(parsed, index=df.index)
            expanded.columns = [f"{col}_{k}" for k in expanded.columns]
            df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    for col in ['cpu_usage_distribution', 'tail_cpu_usage_distribution']:
        if col in df.columns:
            parsed = df[col].apply(parse_distribution)
            df[f'{col}_mean'] = parsed.apply(lambda x: np.mean(x) if x else np.nan)
            df[f'{col}_std']  = parsed.apply(lambda x: np.std(x)  if x else np.nan)
            df.drop(columns=[col], inplace=True)

    if 'failed' in df.columns:
        df['failed'] = (pd.to_numeric(df['failed'], errors='coerce')
                          .fillna(0).astype(int))

    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    df = clean_string_columns(df)

    df.columns = (df.columns
                    .str.strip()
                    .str.encode('ascii', 'ignore')
                    .str.decode('ascii'))

    print(f"  Parsed shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df

# ===========================================================================
# 2. FEATURE ENGINEERING — LEAKAGE-FREE
# ===========================================================================

def engineer_features(df):
    print("\n" + "="*60)
    print("STEP 2: Engineering features (leakage-free)")
    print("="*60)

    if 'start_time' in df.columns and 'end_time' in df.columns:
        df['duration'] = (
            pd.to_numeric(df['end_time'],   errors='coerce') -
            pd.to_numeric(df['start_time'], errors='coerce')
        ).clip(lower=0)
        print("  + duration")

    if 'priority' in df.columns:
        df['priority_tier'] = pd.cut(
            pd.to_numeric(df['priority'], errors='coerce'),
            bins=[-1, 99, 115, 500], labels=[0, 1, 2]
        ).astype(float)
        print("  + priority_tier")

    if 'scheduling_class' in df.columns:
        df['scheduling_class'] = pd.to_numeric(
            df['scheduling_class'], errors='coerce')

    print(f"\n  Total columns: {df.shape[1]}")
    return df

FEATURE_COLS = [
    'resource_request_cpus',
    'resource_request_memory',
    'priority',
    'scheduling_class',
    'collection_type',
    'duration',
    'priority_tier',
    'cycles_per_instruction',
    'memory_accesses_per_instruction',
    # maximum_usage_cpus intentionally NOT here — it is a target
]

def preprocess(df):
    print("\n" + "="*60)
    print("STEP 3: Preprocessing")
    print("="*60)

    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"  Features available: {len(available)}")
    for f in available:
        print(f"    - {f}")

    leakage_columns = {
        'average_usage_cpus', 'average_usage_memory', 'maximum_usage_cpus',
        'maximum_usage_memory', 'cpu_efficiency', 'mem_efficiency',
        'tail_cpu_ratio', 'cpu_burst_ratio', 'cpu_range',
    }
    if leakage_columns.intersection(available):
        raise ValueError(
            "LEAKAGE DETECTED: observed usage feature found in model inputs."
        )
    print("\n  Leakage check PASSED — inputs contain no observed usage values")

    X = df[available].copy()
    X.fillna(X.median(numeric_only=True), inplace=True)

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    splits = {}
    targets = [
        ('average_usage_cpus',   'cpu_avg',  'regression'),
        ('maximum_usage_cpus',   'cpu_peak', 'regression'),
        ('average_usage_memory', 'memory',   'regression'),
        ('failed',               'failure',  'classification'),
    ]

    for target, label, task in targets:
        if target not in df.columns:
            print(f"  Skipping {label} — column not found")
            continue

        y = df[target].copy()

        if task == 'classification':
            y = y.fillna(0).astype(int)
            if y.nunique() < 2:
                continue
            splits[label] = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        else:
            y    = pd.to_numeric(y, errors='coerce')
            mask = y.notna() & (y >= 0)
            if mask.sum() < 10:
                continue
            splits[label] = train_test_split(
                X_scaled[mask], y[mask], test_size=0.2, random_state=42)

        print(f"  Ready: {label}")

    return splits, scaler, available

# ===========================================================================
# 4. TRAINING
# ===========================================================================

def train_regressors(X_train, X_test, y_train, y_test, label):
    print(f"\n  [{label}]")
    records, trained = [], {}
    for name, model in REGRESSORS.items():
        try:
            print(f"    {name}...", end='', flush=True)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mae  = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2   = r2_score(y_test, pred)
            records.append({'Model': name, 'MAE': round(mae, 6),
                            'RMSE': round(rmse, 6), 'R2': round(r2, 4)})
            trained[name] = (model, pred)
            print(f" R2={r2:.4f}")
        except Exception as e:
            print(f" FAILED: {e}")
    return (pd.DataFrame(records).sort_values('R2', ascending=False), trained)

def train_classifiers(X_train, X_test, y_train, y_test):
    print(f"\n  [failure]")
    records, trained = [], {}
    for name, model in CLASSIFIERS.items():
        try:
            print(f"    {name}...", end='', flush=True)
            model.fit(X_train, y_train)
            pred  = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            auc   = (roc_auc_score(y_test, proba)
                     if len(np.unique(y_test)) > 1 else 0.5)
            rep   = classification_report(y_test, pred, output_dict=True)
            records.append({
                'Model':    name,
                'Accuracy': round(rep['accuracy'], 4),
                'F1':       round(rep.get('weighted avg', {})
                                     .get('f1-score', 0), 4),
                'ROC_AUC':  round(auc, 4),
            })
            trained[name] = (model, pred)
            print(f" AUC={auc:.4f}")
        except Exception as e:
            print(f" FAILED: {e}")
    return (pd.DataFrame(records).sort_values('ROC_AUC', ascending=False), trained)

# ===========================================================================
# 5. FIGURES
# All figures: explicit white background, black text, NO raw column names
# ===========================================================================

def _save(fig, filename):
    """Save with guaranteed white background, high DPI, clean edges."""
    fig.savefig(
        filename,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        dpi=300,
        pad_inches=0.2,
    )
    plt.close(fig)
    print(f"  Saved: {filename}")


def fig_line(y_test, pred, label, filename):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             facecolor='white',
                             constrained_layout=True)
    n         = min(300, len(y_test))
    actual    = np.array(y_test)[:n]
    predicted = np.array(pred)[:n]
    idx       = np.arange(n)

    ax = axes[0]
    ax.set_facecolor('white')
    ax.plot(idx, actual,    color='#2563EB', lw=1.2, label='Actual',    alpha=0.85)
    ax.plot(idx, predicted, color='#F97316', lw=1.5, label='Predicted', alpha=0.9)
    ax.set_title(f'Observed vs Forecast: {label}', fontweight='bold', color='black')
    ax.set_ylabel('Usage Fraction', color='black')
    ax.legend()

    ax = axes[1]
    ax.set_facecolor('white')
    error = np.abs(actual - predicted)
    ax.fill_between(idx, 0, error, color='#DC2626', alpha=0.4,
                    label='Absolute Error')
    ax.axhline(error.mean(), color='#DC2626', lw=1.5, linestyle='--',
               label=f'Mean Err = {error.mean():.5f}')
    ax.set_title('Absolute Prediction Error', color='black')
    ax.set_ylabel('|Observed - Forecast|', color='black')
    ax.legend()

    _save(fig, filename)


def fig_bar(df_m, title, metrics, filename):
    n   = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5),
                             facecolor='white',
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric not in df_m.columns:
            continue
        ax.set_facecolor('white')
        vals = df_m[metric]
        bars = ax.barh(df_m['Model'], vals,
                       color=PALETTE[:len(df_m)],
                       edgecolor='none', height=0.6)
        ax.set_xlabel(metric, color='black')
        ax.set_title(metric, color='black')
        ax.invert_yaxis()
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.0005,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8, color='black')

    fig.suptitle(title, fontsize=13, fontweight='bold', color='black')
    _save(fig, filename)


def fig_importance(model, features, label, filename):
    if not hasattr(model, 'feature_importances_'):
        return

    imp = pd.Series(model.feature_importances_, index=features).nlargest(15).sort_values()

    # Replace ALL raw column names with short display labels
    imp.index = [DISPLAY.get(f, f[:8]) for f in imp.index]

    colors = ['#DC2626' if any(k in f for k in ['Burst', 'Range', 'Tail'])
              else '#2563EB' for f in imp.index]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    ax.set_facecolor('white')
    imp.plot(kind='barh', ax=ax, color=colors, edgecolor='none')

    # Title uses NO raw column names
    ax.set_title('Predictor Weights\n(red = spike features)',
                 fontweight='bold', color='black')
    ax.set_xlabel('Relative Importance', color='black')

    _save(fig, filename)


def fig_eda(df, filename):
    cols = [c for c in [
        'resource_request_cpus', 'resource_request_memory',
        'average_usage_cpus',    'average_usage_memory',
        'maximum_usage_cpus',    'cpu_efficiency',
    ] if c in df.columns]
    if not cols:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             facecolor='white',
                             constrained_layout=True)
    axes = axes.flatten()

    for i, col in enumerate(cols[:6]):
        data = df[col].dropna()
        data = data[np.isfinite(data)]
        if len(data) == 0:
            continue
        ax = axes[i]
        ax.set_facecolor('white')
        ax.hist(data, bins=50, color=PALETTE[i % len(PALETTE)],
                edgecolor='none', alpha=0.85)
        # Use short display name — NO raw column names in figure
        ax.set_title(DISPLAY.get(col, col[:12]), color='black')
        ax.set_ylabel('Count', color='black')
        ax.set_xlabel('Value', color='black')

    for j in range(len(cols), 6):
        axes[j].set_visible(False)

    fig.suptitle('Feature Distributions',
                 fontsize=14, fontweight='bold', color='black')
    _save(fig, filename)


def fig_correlation(df, filename):
    key = [c for c in [
        'resource_request_cpus',  'resource_request_memory',
        'average_usage_cpus',     'average_usage_memory',
        'maximum_usage_cpus',
        'cpu_efficiency',         'mem_efficiency',
        'tail_cpu_ratio',         'duration',
        'failed',
    ] if c in df.columns]

    if len(key) < 3:
        return

    corr = df[key].corr()

    # Rename ALL axes labels to short display names — NO raw column names
    labels = [DISPLAY.get(c, c[:10]) for c in key]
    corr.index   = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(11, 9),
                           facecolor='white',
                           constrained_layout=True)
    ax.set_facecolor('white')
    sns.heatmap(
        corr,
        annot=True, fmt='.2f', cmap='coolwarm',
        center=0, square=True, linewidths=0.5,
        ax=ax, annot_kws={'size': 8, 'color': 'black'},
    )
    # Title uses NO raw column names
    ax.set_title(
        'Predictor Correlation Matrix\n'
        '(peak CPU shown for reference only; excluded from model inputs)',
        fontsize=11, fontweight='bold', color='black')
    _save(fig, filename)


def fig_dashboard(results_dict, filename):
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs  = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.4)

    configs = [
        ('cpu_avg',  'R2',      'Avg CPU: R\u00b2',     PALETTE[0], gs[0, 0]),
        ('cpu_peak', 'R2',      'Peak CPU: R\u00b2',    PALETTE[1], gs[0, 1]),
        ('memory',   'R2',      'Memory: R\u00b2',      PALETTE[2], gs[0, 2]),
        ('cpu_avg',  'RMSE',    'Avg CPU: RMSE',        PALETTE[3], gs[1, 0]),
        ('cpu_peak', 'RMSE',    'Peak CPU: RMSE',       PALETTE[4], gs[1, 1]),
        ('failure',  'ROC_AUC', 'Failure: ROC-AUC',    PALETTE[5], gs[1, 2]),
    ]

    for key, metric, title, color, pos in configs:
        ax = fig.add_subplot(pos)
        ax.set_facecolor('white')
        df = results_dict.get(key)

        if df is None or df.empty or metric not in df.columns:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, color='black')
            ax.set_title(title, fontsize=10, color='black')
            continue

        top  = df.nlargest(4, metric)
        bars = ax.barh(top['Model'], top[metric],
                       color=color, edgecolor='none')
        ax.set_title(title, fontsize=10, fontweight='bold', color='black')
        ax.invert_yaxis()

        for bar, val in zip(bars, top[metric]):
            ax.text(bar.get_width() * 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', ha='center',
                    fontsize=7, color='white', fontweight='bold')

    fig.suptitle('Model Evaluation Dashboard',
                 fontsize=14, fontweight='bold', color='black')
    _save(fig, filename)

# ===========================================================================
# 6. MAIN PIPELINE
# ===========================================================================

def run_pipeline(filepath, output_dir='research_paper_outputs_v3'):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  ML RESOURCE UTILIZATION PIPELINE v3")
    print("="*60)

    df = load_and_parse(filepath)
    df = engineer_features(df)

    print("\n" + "="*60)
    print("STEP 4: EDA figures")
    print("="*60)
    fig_eda(df,         f'{output_dir}/fig1_distributions.png')
    fig_correlation(df, f'{output_dir}/fig2_correlation.png')

    splits, scaler, features = preprocess(df)
    all_results = {}

    print("\n" + "="*60)
    print("STEP 5: Training models")
    print("="*60)

    for label in ['cpu_avg', 'cpu_peak', 'memory']:
        if label not in splits:
            continue
        Xtr, Xte, ytr, yte = splits[label]

        metrics, trained      = train_regressors(Xtr, Xte, ytr, yte, label)
        all_results[label]    = metrics
        best_name             = metrics.iloc[0]['Model']
        best_model, best_pred = trained[best_name]

        fig_line(
            yte, best_pred,
            label,
            f'{output_dir}/fig_{label}_actual_vs_predicted.png'
        )
        fig_bar(
            metrics,
            f'{label} Prediction',
            ['MAE', 'RMSE', 'R2'],
            f'{output_dir}/fig_{label}_comparison.png'
        )
        fig_importance(
            best_model, features, label,
            f'{output_dir}/fig_{label}_importance.png'
        )
        joblib.dump(best_model, f'{output_dir}/best_{label}_model.pkl')

    if 'failure' in splits:
        Xtr, Xte, ytr, yte    = splits['failure']
        metrics, trained       = train_classifiers(Xtr, Xte, ytr, yte)
        all_results['failure'] = metrics
        best_name              = metrics.iloc[0]['Model']
        joblib.dump(
            trained[best_name][0],
            f'{output_dir}/best_failure_model.pkl'
        )
        fig_bar(
            metrics,
            'Failure Prediction',
            ['Accuracy', 'F1', 'ROC_AUC'],
            f'{output_dir}/fig_failure_comparison.png'
        )

    fig_dashboard(all_results, f'{output_dir}/fig_dashboard.png')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)

    for label, df_m in all_results.items():
        print(f"\n  [{label.upper()}]")
        print(df_m.to_string(index=False))

    return all_results

# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == '__main__':
    CSV_PATH   = "dataset/cpu_data.csv"   # <- CHANGE THIS TO YOUR FILE
    OUTPUT_DIR = "research_paper_outputs_v3"
    run_pipeline(CSV_PATH, OUTPUT_DIR)