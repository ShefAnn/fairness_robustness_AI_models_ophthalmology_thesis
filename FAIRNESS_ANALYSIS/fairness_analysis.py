#LLM Disclaimer: Debugging was done with the help of ChatGPT: https://chatgpt.com/

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def safe_confusion(y_true, y_pred):
    # to avoid an invalid confusion matrix
    if len(np.unique(y_true)) < 2:
        return None
    return confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()


def safe_specificity(y_true, y_pred):
    # to avoid an invalid specificity value if confusion matrix is None
    cm = safe_confusion(y_true, y_pred)
    if cm is None:
        return np.nan
    tn, fp, _, _ = cm
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def safe_rates(y_true, y_pred):
    # to avoid invalid rates values if confusion matrix is None
    cm = safe_confusion(y_true, y_pred)
    if cm is None:
        return np.nan, np.nan
    tn, fp, fn, tp = cm
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return fpr, tpr


def bootstrap_values_and_gaps(
    df,
    group_col,
    metric,
    n_boot=2000,
):
    """
    bootstrap the dataset to estimate a metric’s mean, median and 95% CI, overall and per group
    compute the performance gap between groups and the worst-performing group
    """
    group_names = sorted(df[group_col].dropna().unique())

    values_for_fold = []
    values_for_groups = []
    gaps = []
    worst_group_vals = []

    for _ in range(n_boot):
        sample = df.sample(frac=1, replace=True)

        try:
            if metric == 'accuracy':
                overall = accuracy_score(sample['label'], sample['pred'])

            elif metric == 'f1':
                overall = f1_score(sample['label'], sample['pred'], zero_division=0)

            elif metric == 'precision':
                overall = precision_score(sample['label'], sample['pred'], zero_division=0)

            elif metric in ['recall', 'sensitivity']:
                overall = recall_score(sample['label'], sample['pred'], zero_division=0)

            elif metric == 'bal_acc':
                overall = balanced_accuracy_score(sample['label'], sample['pred'])

            elif metric == 'auc':
                overall = roc_auc_score(
                    sample['label'],
                    sample['probs'] )

            elif metric == 'specificity':
                overall = safe_specificity(sample['label'], sample['pred'])

            else:
                raise ValueError(metric)

        except ValueError:
            overall = np.nan

        values_for_fold.append(overall)

        group_vals = {}

        for g in group_names:
            gdf = sample[sample[group_col] == g]

            try:
                if metric == 'accuracy':
                    val = accuracy_score(gdf['label'], gdf['pred'])

                elif metric == 'f1':
                    val = f1_score(gdf['label'], gdf['pred'], zero_division=0)

                elif metric == 'precision':
                    val = precision_score(gdf['label'], gdf['pred'], zero_division=0)

                elif metric in ['recall', 'sensitivity']:
                    val = recall_score(gdf['label'], gdf['pred'], zero_division=0)

                elif metric == 'bal_acc':
                    val = balanced_accuracy_score(gdf['label'], gdf['pred'])

                elif metric == 'auc':
                    val = roc_auc_score(
                        gdf['label'],
                        gdf['probs']
                    )

                elif metric == 'specificity':
                    val = safe_specificity(gdf['label'], gdf['pred'])

                else:
                    val = np.nan

            except ValueError:
                val = np.nan

            group_vals[g] = val

        values_for_groups.append(group_vals)

        vals = list(group_vals.values())
        if np.all(np.isnan(vals)):
            gaps.append(np.nan)
            worst_group_vals.append(np.nan)
        else:
            gaps.append(np.nanmax(vals) - np.nanmin(vals))
            worst_group_vals.append(np.nanmin(vals))

    group_cis = {}
    for g in group_names:
        g_vals = [
            d[g] for d in values_for_groups
            if g in d and not np.isnan(d[g])
        ]
        if len(g_vals) == 0:
            group_cis[g] = np.array([np.nan, np.nan, np.nan])
        else:
            group_cis[g] = np.percentile(g_vals, [2.5, 50, 97.5])

    return {
        'overall_median': np.nanmedian(values_for_fold),
        'overall_mean': np.nanmean(values_for_fold),
        'overall_ci': np.percentile(values_for_fold, [2.5, 50, 97.5]),

        'group_medians': {
            g: np.nanmedian([d[g] for d in values_for_groups])
            for g in group_names
        },
        'group_means': {
            g: np.nanmean([d[g] for d in values_for_groups])
            for g in group_names
        },
        'group_cis': group_cis,

        'gap_median': np.nanmedian(gaps),
        'gap_mean': np.nanmean(gaps),
        'gap_ci': np.percentile(gaps, [2.5, 50, 97.5]),

        'worst_group_median': np.nanmedian(worst_group_vals),
        'worst_group_mean': np.nanmean(worst_group_vals),
        'worst_group_ci': np.percentile(worst_group_vals, [2.5, 50, 97.5])
    }


def fairness_bootstrap(df, group_col='age_group', n_boot=2000):
    # bootstrap for EO(d_TPR), AOD, d_FPR
    delta_fprs = []
    delta_tprs = []
    aods = []

    for _ in range(n_boot):
        sample = df.sample(frac=1, replace=True)
        fprs, tprs = [], []

        for _, gdf in sample.groupby(group_col):
            fpr, tpr = safe_rates(gdf['label'], gdf['pred'])
            fprs.append(fpr)
            tprs.append(tpr)

        delta_fpr = np.nanmax(fprs) - np.nanmin(fprs)
        delta_tpr = np.nanmax(tprs) - np.nanmin(tprs)

        delta_fprs.append(delta_fpr)
        delta_tprs.append(delta_tpr)
        aods.append((delta_fpr + delta_tpr) / 2)

    return {
        'delta_fpr_ci': np.percentile(delta_fprs, [2.5, 50, 97.5]),
        'delta_fpr_mean': np.nanmean(delta_fprs),
        'eo_ci': np.percentile(delta_tprs, [2.5, 50, 97.5]),
        'eo_mean': np.nanmean(delta_tprs), 
        'aod_ci': np.percentile(aods, [2.5, 50, 97.5]),
        'aod_mean': np.nanmean(aods)   
    }



def do_bootstrap_for_metrics_and_fairness_binary(
    folds_dfs,
    group_col='age_group',
    logits=False,
    probs= False,
    n_boot=2000
):
    """
    apply bootstrap for particular group
    """
    all_metrics = {}

    all_data_df = pd.concat(folds_dfs.values(), ignore_index=True)
    if logits:
        all_data_df['probs'] = sigmoid(all_data_df['logits'])
    if probs:
            all_data_df['probs'] = all_data_df['probs']

    for fold, df in folds_dfs.items():
        df = df.copy()
        if logits:
            df['probs'] = sigmoid(df['logits'])
        if probs:
            df['probs'] = df['probs']

        metrics = {}
        metrics_list = ['accuracy','sensitivity','specificity','precision','f1','bal_acc']
        if logits:
            metrics_list.append('auc')
        if probs:
            metrics_list.append('auc')

        for metric in metrics_list:
            metrics[metric] = bootstrap_values_and_gaps(
                df,
                group_col=group_col,
                metric=metric,
                n_boot=n_boot
            )

        metrics.update(
            fairness_bootstrap(df, group_col=group_col, n_boot=n_boot)
        )

        all_metrics[fold] = metrics

    metrics_all = {}
    for metric in metrics_list:
        metrics_all[metric] = bootstrap_values_and_gaps(
            all_data_df,
            group_col=group_col,
            metric=metric,
            n_boot=n_boot
        )

    metrics_all.update(
        fairness_bootstrap(all_data_df, group_col=group_col, n_boot=n_boot)
    )

    all_metrics['all_data'] = metrics_all

    return all_metrics


###################################################################################################################################
# MULTICLASS
###################################################################################################################################
def bootstrap_values_and_gaps_multiclass(
    df,
    group_col,
    metric,
    n_boot=2000
):
    group_names = sorted(df[group_col].dropna().unique())

    values_for_fold = []
    values_for_groups = []
    gaps = []
    worst_group_vals = []

    for _ in range(n_boot):
        sample = df.sample(frac=1, replace=True)

        try:
            if metric == 'accuracy':
                overall = accuracy_score(sample['label'], sample['pred'])
            elif metric == 'f1':
                overall = f1_score(sample['label'], sample['pred'], average='macro', zero_division=0)
            elif metric == 'precision':
                overall = precision_score(sample['label'], sample['pred'], average='macro', zero_division=0)
            elif metric in ['recall', 'sensitivity']:
                overall = recall_score(sample['label'], sample['pred'], average='macro', zero_division=0)
            elif metric == 'bal_acc':
                overall = balanced_accuracy_score(sample['label'], sample['pred'])
            else:
                overall = np.nan
        except:
            overall = np.nan

        values_for_fold.append(overall)

        group_vals = {}
        for g in group_names:
            gdf = sample[sample[group_col] == g]
            try:
                if metric == 'accuracy':
                    val = accuracy_score(gdf['label'], gdf['pred'])
                elif metric == 'f1':
                    val = f1_score(gdf['label'], gdf['pred'], average='macro', zero_division=0)
                elif metric == 'precision':
                    val = precision_score(gdf['label'], gdf['pred'], average='macro', zero_division=0)
                elif metric in ['recall', 'sensitivity']:
                    val = recall_score(gdf['label'], gdf['pred'], average='macro', zero_division=0)
                elif metric == 'bal_acc':
                    val = balanced_accuracy_score(gdf['label'], gdf['pred'])
                else:
                    val = np.nan
            except:
                val = np.nan
            group_vals[g] = val

        values_for_groups.append(group_vals)

        vals = list(group_vals.values())
        if np.all(np.isnan(vals)):
            gaps.append(np.nan)
            worst_group_vals.append(np.nan)
        else:
            gaps.append(np.nanmax(vals) - np.nanmin(vals))
            worst_group_vals.append(np.nanmin(vals))

    group_cis = {}
    for g in group_names:
        g_vals = [d[g] for d in values_for_groups if g in d and not np.isnan(d[g])]
        if len(g_vals) == 0:
            group_cis[g] = np.array([np.nan, np.nan, np.nan])
        else:
            group_cis[g] = np.percentile(g_vals, [2.5, 50, 97.5])

    return {
        'overall_median': np.nanmedian(values_for_fold),
        'overall_mean': np.nanmean(values_for_fold),
        'overall_ci': np.percentile(values_for_fold, [2.5, 50, 97.5]),
        'group_medians': {g: np.nanmedian([d[g] for d in values_for_groups]) for g in group_names},
        'group_means': {g: np.nanmean([d[g] for d in values_for_groups]) for g in group_names},
        'group_cis': group_cis,
        'gap_median': np.nanmedian(gaps),
        'gap_mean': np.nanmean(gaps),
        'gap_ci': np.percentile(gaps, [2.5, 50, 97.5]),
        'worst_group_median': np.nanmedian(worst_group_vals),
        'worst_group_mean': np.nanmean(worst_group_vals),
        'worst_group_ci': np.percentile(worst_group_vals, [2.5, 50, 97.5])
    }

def do_bootstrap_for_metrics_and_fairness_multiclass(
    folds_dfs,
    group_col='age_group',
    logits=False,
    n_boot=2000
):
    all_metrics = {}

    all_data_df = pd.concat(folds_dfs.values(), ignore_index=True)
    if logits:
        all_data_df['probs'] = sigmoid(all_data_df['logits'])

    for fold, df in folds_dfs.items():
        df = df.copy()
        if logits:
            df['probs'] = sigmoid(df['logits'])

        metrics = {}
        metrics_list = ['accuracy','sensitivity','specificity','precision','f1','bal_acc']

        for metric in metrics_list:
            metrics[metric] = bootstrap_values_and_gaps(
                df,
                group_col=group_col,
                metric=metric,
                n_boot=n_boot
            )

        metrics.update(
            fairness_bootstrap(df, group_col=group_col, n_boot=n_boot)
        )

        all_metrics[fold] = metrics

    metrics_all = {}
    for metric in metrics_list:
        metrics_all[metric] = bootstrap_values_and_gaps(
            all_data_df,
            group_col=group_col,
            metric=metric,
            n_boot=n_boot
        )

    metrics_all.update(
        fairness_bootstrap(all_data_df, group_col=group_col, n_boot=n_boot)
    )

    all_metrics['all_data'] = metrics_all

    return all_metrics 
    
###############################################################################################################
# PLOTS
###############################################################################################################
def get_metrics_after_bootstrap(data):
    rows_groups = []
    for split_name, split_data in data.items():
        split_type = "all_data" if split_name == "all_data" else "fold"
        
        for metric, metric_data in split_data.items():
            if not isinstance(metric_data, dict):
                continue
            if "group_medians" not in metric_data:
                continue
            for group, median in metric_data["group_medians"].items():
                ci = metric_data["group_cis"].get(group, [np.nan] * 3)
                mean = metric_data.get("group_means", {}).get(group, np.nan)
                
                rows_groups.append({
                    "split_type": split_type,
                    "split": split_name,
                    "metric": metric,
                    "group": group,
                    "median": median,
                    "mean": mean, 
                    "ci_low": ci[0],
                    "ci_mid": ci[1],
                    "ci_high": ci[2],
                })
    df_groups = pd.DataFrame(rows_groups)

    rows_overall = []

    for split_name, split_data in data.items():
        split_type = "all_data" if split_name == "all_data" else "fold"

        for metric, metric_data in split_data.items():

            if not isinstance(metric_data, dict):
                continue
            if "overall_median" not in metric_data:
                continue

            ci = metric_data.get("overall_ci", [np.nan, np.nan, np.nan])

            rows_overall.append({
                "split_type": split_type,
                "split": split_name,
                "metric": metric,

                "overall_median": metric_data["overall_median"],
                "overall_mean": metric_data["overall_mean"],
                "ci_low": ci[0],
                "ci_mid": ci[1],
                "ci_high": ci[2],

                "gap_median": metric_data.get("gap_median", np.nan),
                "gap_mean": metric_data.get("gap_mean", np.nan),
                "worst_group_median": metric_data.get("worst_group_median", np.nan),
                "worst_group_mean": metric_data.get("worst_group_mean", np.nan),
            })

    df_overall = pd.DataFrame(rows_overall)


    rows_fairness = []

    for split_name, split_data in data.items():
        eo_mean = split_data.get("eo_mean", np.nan)
        aod_mean = split_data.get("aod_mean", np.nan)
        delta_fpr_mean = split_data.get("delta_fpr_mean", np.nan)
        rows_fairness.append({
            "split": split_name,
            "eo_ci_low": split_data.get("eo_ci", [np.nan]*3)[0],
            "eo_ci_mid": split_data.get("eo_ci", [np.nan]*3)[1],
            "eo_ci_high": split_data.get("eo_ci", [np.nan]*3)[2],
            "eo_mean": eo_mean,  
            "aod_ci_low": split_data.get("aod_ci", [np.nan]*3)[0],
            "aod_ci_mid": split_data.get("aod_ci", [np.nan]*3)[1],
            "aod_ci_high": split_data.get("aod_ci", [np.nan]*3)[2],
            "aod_mean": aod_mean,   
            "delta_fpr_ci_low": split_data.get("delta_fpr_ci", [np.nan]*3)[0],
            "delta_fpr_ci_mid": split_data.get("delta_fpr_ci", [np.nan]*3)[1],
            "delta_fpr_ci_high": split_data.get("delta_fpr_ci", [np.nan]*3)[2],
            "delta_fpr_mean": delta_fpr_mean,
        })

    df_fairness = pd.DataFrame(rows_fairness)
    return df_groups, df_overall, df_fairness

    
def plot_medians_CIs_for_groups_whole_dataset(df_groups, groups="age-groups"):
    metrics = df_groups["metric"].unique()

    plt.figure(figsize=(14, 9))

    for i, metric in enumerate(metrics, 1):
        ax = plt.subplot(3, 3, i)

        df_all = (
            df_groups[
                (df_groups["metric"] == metric) &
                (df_groups["split_type"] == "all_data")
            ]
            .sort_values("group")
        )

        if df_all.empty:
            continue

        ax.errorbar(
            df_all["group"],
            df_all["median"],
            yerr=[
                df_all["median"] - df_all["ci_low"],
                df_all["ci_high"] - df_all["median"]
            ],
            fmt="o",
            capsize=4,
            color="black",
            markersize=6,
            linestyle="none",
            label="Median"
        )

        if "mean" in df_all.columns:
            ax.scatter(
                df_all["group"],
                df_all["mean"],
                color="red",
                marker="x",
                s=80,
                alpha=0.5,
                label="Mean"
            )

        ax.set_ylim(0, 1)
        ax.set_title(f"Metric: {metric}, Groups: {groups}")
        ax.set_xlabel(groups)
        ax.set_ylabel("metric value")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.show()




def plot_metrics_folds_all_data_and_worst_group_median(df_overall, groups="age-groups"):
    metrics = df_overall["metric"].unique()
    folds = [f for f in df_overall["split"].unique() if f != "all_data"]
    colors = plt.cm.tab10.colors

    n_metrics = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    worst_marker = plt.Line2D([0], [0], color='black', marker='s', linestyle='', markersize=8)
    legend_handles.append(worst_marker)
    legend_labels.append("Worst group median")

    mean_marker = plt.Line2D([0], [0], color='black', marker='x', linestyle='', markersize=6)
    legend_handles.append(mean_marker)
    legend_labels.append("Mean")

    short_labels = [f"F{i}" for i in range(len(folds))] + ["All"]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for j, fold in enumerate(folds):
            row = df_overall[(df_overall["split"] == fold) & (df_overall["metric"] == metric)].iloc[0]

            yerr = np.array([
                [row["overall_median"] - row["ci_low"]],
                [row["ci_high"] - row["overall_median"]]
            ])

            handle = ax.errorbar(
                x=[j],
                y=[row["overall_median"]],
                yerr=yerr,
                fmt="o",
                capsize=5,
                color=colors[j % len(colors)]
            )

            if i == 0:
                legend_handles.append(handle)
                legend_labels.append(fold)

            ax.scatter(
                [j],
                [row["worst_group_median"]],
                color=colors[j % len(colors)],
                marker='s',
                s=80
            )

            if "overall_mean" in row:
                ax.scatter(
                    [j],
                    [row["overall_mean"]],
                    color=colors[j % len(colors)],
                    marker='x',
                    s=80
                )

        if "all_data" in df_overall["split"].values:
            row_all = df_overall[(df_overall["split"] == "all_data") & (df_overall["metric"] == metric)].iloc[0]

            yerr_all = np.array([
                [row_all["overall_median"] - row_all["ci_low"]],
                [row_all["ci_high"] - row_all["overall_median"]]
            ])

            ax.errorbar(
                x=[len(folds)],
                y=[row_all["overall_median"]],
                yerr=yerr_all,
                fmt="o",
                capsize=5,
                color='black'
            )

            ax.scatter(
                [len(folds)],
                [row_all["worst_group_median"]],
                color='black',
                marker='s',
                s=80
            )

            if "overall_mean" in row_all:
                ax.scatter(
                    [len(folds)],
                    [row_all["overall_mean"]],
                    color='black',
                    marker='x',
                    alpha=0.5,
                    s=80
                )

        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(folds) + 1))
        ax.set_xticklabels(short_labels, rotation=45)
        ax.set_ylabel("Score")
        ax.grid(True, linestyle='--', alpha=0.6)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Overall performance with Worst Group per metric(sq) and Mean(X), Groups: {groups}", fontsize=16)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    n_rows_legend = 2
    n_items = len(legend_handles)
    ncol_legend = math.ceil(n_items / n_rows_legend)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=ncol_legend,
        title="Legend",
        frameon=False
    )

    plt.show()


def plot_fairness_metrics(df_fairness, groups="age-groups"):
    metrics = ["eo", "aod", "delta_fpr"]
    colors = plt.cm.tab10.colors

    folds = df_fairness["split"].values

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))

    legend_handles = []
    legend_labels = []

    marker_mid = "o"

    mean_marker = plt.Line2D([0], [0], color='black', marker='x', linestyle='', markersize=6)
    legend_handles.append(mean_marker)
    legend_labels.append("Mean")

    short_names = [f"F{i}" if f.startswith("fold") else "ALL" for i, f in enumerate(folds)]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, fold in enumerate(folds):
            row = df_fairness[df_fairness["split"] == fold].iloc[0]

            y_mid = row[f"{metric}_ci_mid"]
            y_low = row[f"{metric}_ci_low"]
            y_high = row[f"{metric}_ci_high"]

            color = 'black' if fold == "all_data" else colors[j % len(colors)]

            handle = ax.errorbar(
                x=[j],
                y=[y_mid],
                yerr=[[y_mid - y_low], [y_high - y_mid]],
                fmt=marker_mid,
                capsize=5,
                color=color
            )

            mean_value = row.get(f"{metric}_mean", None)
            if mean_value is not None:
                ax.scatter(
                    [j],
                    [mean_value],
                    color=color,
                    marker='x',
                    s=80,
                    alpha=0.5
                )

            if i == 0:
                legend_handles.append(handle)
                legend_labels.append(short_names[j])

        ax.set_title(metric.upper())
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(folds)))
        ax.set_xticklabels(short_names, rotation=45)
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    n_rows_legend = 2
    n_items = len(legend_handles)
    ncol_legend = math.ceil(n_items / n_rows_legend)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=ncol_legend,
        title="Folds / Mean",
        frameon=False
    )

    fig.suptitle(f"Fairness metrics per fold + all_data, Groups: {groups}", fontsize=16)
    plt.show()
