# -------------------------
# Plotting utilities
# -------------------------
import contextlib
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    from statannotations.Annotator import Annotator
except Exception:
    Annotator = None

# set up logger with logging package
logger = logging.getLogger(__name__)


def plot_feature_matrix(feature_matrix, x_labels, ax=None, fig_size=(10, 2)):
    """
    Plot a binary feature matrix below a boxplot using scatter plot markers.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    x_positions = list(range(len(x_labels)))
    feature_matrix = feature_matrix.loc[x_labels]  # ensure order matches

    features = feature_matrix.columns

    for i, feature in enumerate(features):
        y = [len(features) - i - 1] * len(x_labels)  # Flip Y order
        values = feature_matrix[feature].tolist()
        ax.scatter(
            x_positions,
            y,
            c=values,
            cmap="Greys",
            vmin=0,
            vmax=1,  # force full contrast
            marker="o",
            s=100,
            edgecolor="black",
        )

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features[::-1])  # Reverse to match flipped Y
    ax.set_xticks([])  # Hide x-axis labels
    ax.set_ylim(-0.5, len(features) - 0.5)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    return ax


def make_dataset_order(df, dataset_order=None, group_col="datasets"):
    if dataset_order is None:
        return list(df[group_col].unique())
    # keep only datasets present
    present = set(df[group_col].unique())
    return [d for d in dataset_order if d in present]


def add_background_bands(ax_list, dataset_order, alpha=0.18, color="lightgrey"):
    for i, _ in enumerate(dataset_order):
        if i % 2 == 0:
            for ax in ax_list:
                ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=alpha, zorder=0)


def build_model_legend_handles(
    model_order, model_labels, model_color_map, include_random=False
):
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=model_color_map[m],
            markersize=8,
            label=model_labels.get(m, m),
        )
        for m in model_order
    ]
    if include_random:
        handles.append(
            Line2D([0], [0], color="grey", lw=1.5, linestyle="--", label="Random")
        )
    return handles


def build_pairs_by_dataset(dataset_order, comparisons):
    pairs = []
    for ds in dataset_order:
        for ma, mb in comparisons:
            pairs.append(((ds, ma), (ds, mb)))
    return pairs


def format_p_or_q(sig_val, style: str = "stars") -> str:
    """
    Format a p/q value using one of several styles.

    Legend:
      ns: p <= 1.00e+00
       *: 1.00e-02 < p <= 5.00e-02
      **: 1.00e-03 < p <= 1.00e-02
     ***: 1.00e-04 < p <= 1.00e-03
    ****: p <= 1.00e-04

    Returns:
      "ns", "*", "**", "***", "****"
    """
    if sig_val is None:
        return "ns"
    try:
        p = float(sig_val)
    except (TypeError, ValueError):
        return "ns"
    if np.isnan(p):
        return "ns"

    # clamp
    if p < 0:
        p = 0.0
    if p > 1:
        p = 1.0

    style = style.lower().strip()

    if style == "stars":
        if p <= 1e-4:
            return "****"
        elif p <= 1e-3:
            return "***"
        elif p <= 1e-2:
            return "**"
        elif p <= 5e-2:
            return "*"
        else:
            return "ns"

    if style == "threshold_float":
        # match your star bins, but return the bin label instead
        if p <= 1e-4:
            return "0.0001"
        if p <= 1e-3:
            return "0.001"
        if p <= 1e-2:
            return "0.01"
        if p <= 5e-2:
            return "0.05"
        return "ns"

    if style == "threshold_sci":
        if p <= 1e-4:
            return "1e-4"
        if p <= 1e-3:
            return "1e-3"
        if p <= 1e-2:
            return "1e-2"
        if p <= 5e-2:
            return "0.05"
        return "ns"

    if style == "exact_3sig":
        # 3 significant digits, switches to scientific automatically when small
        # examples: 0.05 -> "0.05", 0.0123 -> "0.0123", 1.2e-05 -> "1.2e-05"
        return f"{p:.3g}"

    raise ValueError(
        f"Unknown style={style!r}. Choose from: "
        "'stars', 'threshold_float', 'threshold_sci', 'exact_3sig'."
    )


def pick_sig_value(row, p_col="p_wilcoxon", q_col="q_fdr_bh", use_q=True):
    """Return (sig_val, sig_name) preferring q when requested and available."""
    p = row.get(p_col, np.nan)
    q = row.get(q_col, np.nan) if q_col in getattr(row, "index", []) else np.nan

    has_q = not (q is None or (isinstance(q, float) and np.isnan(q)))
    if use_q and has_q:
        return float(q), "q"
    return float(p) if p is not None else np.nan, "p"


def _format_comparison_text(
    row,
    delta_col,
    annot_style,
    use_q=True,
    just_significance=False,
    no_sig_name=True,
):
    """Return a compact label for a df_dset_comp row."""
    delta = float(row.get(delta_col, np.nan))

    # choose significance value
    sig_val, sig_name = pick_sig_value(
        row, p_col="p_wilcoxon", q_col="q_fdr_bh", use_q=use_q
    )

    stars = format_p_or_q(sig_val, style=annot_style)
    if just_significance and no_sig_name:
        if stars == "ns":
            return "ns"
        else:
            return f"{stars}"
    elif just_significance:
        if stars == "ns":
            return f"{sig_name}=ns"
        else:
            return f"{sig_name}≤{stars}"
    elif no_sig_name:
        # format: +0.03 (**) or +0.03 (ns)
        if stars == "ns":
            return f"{delta:+.2f} (ns)"
        else:
            return f"{delta:+.2f} ({stars})"
    else:
        # format: +0.03 (q≤**) or +0.03 (p≤ns)
        if stars == "ns":
            return f"{delta:+.2f} ({sig_name}=ns)"
        else:
            return f"{delta:+.2f} ({sig_name}≤{stars})"


def style_pointplot_medians(
    ax, start_line_idx, color="lightgray", zorder=6, markersize=14, linewidth=1.0
):
    # only restyle lines added after pointplot call
    for line in ax.lines[start_line_idx:]:
        line.set_color(color)
        line.set_markeredgecolor(color)
        line.set_markerfacecolor(color)
        line.set_zorder(zorder)
        line.set_markersize(markersize)  # wider underscore
        line.set_linewidth(linewidth)  # thickness


def set_stripplot_face_edge_alpha(
    ax,
    face_alpha=0.3,
    edge_alpha=1.0,
    linewidth=0.8,
    # zorder=2,
    start_collection_idx=None,
):
    """
    Post-process seaborn stripplot PathCollections so that
    - marker fill is semi-transparent
    - marker edge is fully opaque
    - edge color matches fill color

    Parameters
    ----------
    ax : matplotlib Axes
        Axis containing the stripplot.
    face_alpha : float
        Alpha for marker face (e.g. 0.3).
    edge_alpha : float
        Alpha for marker edge (e.g. 1.0).
    linewidth : float
        Line width for marker edges.
    zorder : float
        Z-order for the points.
    """
    for coll in ax.collections[start_collection_idx:]:
        # IMPORTANT: remove any collection-wide alpha so RGBA alphas take effect
        coll.set_alpha(face_alpha)
        fcs = coll.get_facecolors()
        ecs = coll.get_edgecolors()

        if len(fcs):
            fcs[:, 3] = face_alpha
            coll.set_facecolors(fcs)

        if len(ecs):
            ecs[:, 3] = edge_alpha
            coll.set_edgecolors(ecs)

        coll.set_linewidths(linewidth)
        # coll.set_zorder(zorder)


def get_valid_group_order(df, group_order, group_col="datasets"):
    """
    Return a filtered dataset order that includes only the group_col values present in df.
    """
    available = df[group_col].unique().tolist()
    return [d for d in group_order if d in available]


####################
# Functions for P1000 empirical figures
####################


def plot_boxplot(
    data,
    x_name="datasets",
    y_name="value",
    color_map=None,
    ax=None,
    no_x_labels=False,
    showfliers=True,
    dataset_order=None,
):
    """
    Plot a seaborn boxplot with optional custom color mapping.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    valid_order = (
        get_valid_group_order(data, dataset_order, group_col=x_name)
        if dataset_order
        else None
    )
    sns.boxplot(
        data=data,
        x=x_name,
        y=y_name,
        ax=ax,
        showfliers=showfliers,
        palette=color_map if color_map else None,
        order=valid_order,
    )
    if not showfliers:
        sns.stripplot(
            data=data,
            x=x_name,
            y=y_name,
            ax=ax,
            color="black",
            jitter=0.2,
            alpha=0.3,
            order=valid_order,
        )
    if no_x_labels:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.tick_params(axis="x", labelbottom=True)
        plt.setp(ax.get_xticklabels(), rotation=90)
    return ax


def plot_grouped_stripplot_with_medians_and_feature_matrix(
    df,
    metric_col,
    pair_col,
    model_order,
    model_labels,
    model_color_map,
    random_performance_value=None,
    dataset_order=None,
    feature_matrix=None,
    plot_feature_matrix_func=None,
    df_model_comp=None,
    comparisons=(("pnet", "rf"), ("pnet", "logistic_regression")),
    title=None,
    figsize=(10, 6.0),
    height_ratios=(6, 2),
    exclude_points_models=(),
    annotator_loc="inside",
    annotator_line_offset=0.03,
    annotator_text_offset=2,  # points
    annot_style="threshold_float",
    annot_just_significance=False,
    annot_no_sig_name=False,
    alpha=0.05,
    legend_outside=True,
    save_legend_path=None,
    save_legend_kwargs=None,
    draw_legend=True,
    show_boxplot=True,
    boxplot_kwargs=None,
    show_medians=False,
    show_stripplot=True,
    strip_face_alpha=0.6,
    strip_edge_alpha=1.0,
    strip_size=1,
    y_label=None,
):
    """
    Used to generate the dataset-grouped AUPRC plots for P1000 results.
    Expects df columns:
      - 'datasets' (x)
      - 'model_type' (hue)
      - metric_col (y)
      - pair_col (for upstream stats; not used directly unless you compute df_model_comp elsewhere)
    """

    dataset_order = make_dataset_order(df, dataset_order, group_col="datasets")

    fig, axes = plt.subplots(
        2 if feature_matrix is not None else 1,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios}
        if feature_matrix is not None
        else None,
        sharex=True if feature_matrix is not None else False,
    )

    if feature_matrix is not None:
        ax_main, ax_matrix = axes
    else:
        ax_main = axes
        ax_matrix = None

    # background bands
    ax_list = [ax_main] + ([ax_matrix] if ax_matrix is not None else [])
    add_background_bands(ax_list, dataset_order)

    # 1) Boxplot (optional)
    if show_boxplot:
        _boxplot_kwargs = dict(
            dodge=True, ax=ax_main, showfliers=False, boxprops={"alpha": 0.3}
        )
        if boxplot_kwargs:
            _boxplot_kwargs.update(boxplot_kwargs)
        sns.boxplot(
            data=df,
            x="datasets",
            y=metric_col,
            hue="model_type",
            order=dataset_order,
            hue_order=model_order,
            palette=model_color_map,
            **_boxplot_kwargs,
        )

    # stripplot points (optionally exclude some models)
    df_points = df[~df["model_type"].isin(exclude_points_models)].copy()

    if show_stripplot:
        # record collections so you can post-style if you want
        start_idx = len(ax_main.collections)
        sns.stripplot(
            data=df_points,
            x="datasets",
            y=metric_col,
            hue="model_type",
            order=dataset_order,
            hue_order=model_order,
            palette=model_color_map,
            dodge=True,
            jitter=True,
            size=strip_size,
            ax=ax_main,
            edgecolor="face",
            linewidth=0.8,  # we'll restyle after
            alpha=strip_face_alpha,
        )

        with contextlib.suppress(NameError):
            set_stripplot_face_edge_alpha(
                ax_main,
                face_alpha=strip_face_alpha,
                edge_alpha=strip_edge_alpha,
                start_collection_idx=start_idx,
            )

        # remove seaborn legend here (we'll add custom)
        ax_main.legend([], [], frameon=False)

    # medians via pointplot (per dataset x model)
    if show_medians:
        sns.pointplot(
            data=df,
            x="datasets",
            y=metric_col,
            hue="model_type",
            order=dataset_order,
            hue_order=model_order,
            estimator=np.median,
            errorbar=None,
            linestyle="none",
            dodge=0.5,
            markers="_",
            markersize=20,
            ax=ax_main,
        )

    # random baseline
    if random_performance_value is not None:
        ax_main.axhline(
            random_performance_value, linestyle="--", color="grey", zorder=1
        )

    # cosmetics
    ax_main.set_title(title if title is not None else metric_col)
    ax_main.set_ylabel(metric_col)
    ax_main.set_xlabel("")
    ax_main.set_xticklabels(dataset_order, rotation=90)
    yticks = ax_main.get_yticks()
    fixed_yticks = yticks[yticks <= 1.0]
    ax_main.set_yticks(fixed_yticks)
    ax_main.grid(axis="y", alpha=0.18)
    ax_main.set_axisbelow(True)
    if y_label is not None:
        ax_main.set_ylabel(y_label)

    # legend
    handles = build_model_legend_handles(
        model_order=model_order,
        model_labels=model_labels,
        model_color_map=model_color_map,
        include_random=(random_performance_value is not None),
    )
    labels = [h.get_label() for h in handles]

    # optionally save a standalone legend
    if save_legend_path is not None:
        _kw = dict(title="Model", ncol=3, figsize=(5.5, 0.8), dpi=300)
        if save_legend_kwargs is not None:
            _kw.update(save_legend_kwargs)
        save_legend_only(handles, labels, save_legend_path, **_kw)

    # optionally draw legend on the main plot
    if draw_legend:
        if legend_outside:
            ax_main.legend(
                handles=handles,
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                title="Model",
            )
        else:
            ax_main.legend(handles=handles, loc="best", title="Model")

    # feature matrix
    if feature_matrix is not None:
        if plot_feature_matrix_func is None:
            raise ValueError(
                "plot_feature_matrix_func must be provided when feature_matrix is not None."
            )
        plot_feature_matrix_func(feature_matrix, x_labels=dataset_order, ax=ax_matrix)

    # annotations (optional) -- MODEL-MODEL across datasets
    # For each dataset ds, find all rows in df_model_comp where datasets == ds,
    # format each row using _format_comparison_text,
    # and place the annotation above ds.
    # annotations (optional)
    annotator = None
    if df_model_comp is not None:
        if Annotator is None:
            raise ImportError(
                "statannotations is not installed/available, but df_model_comp was provided."
            )

        delta_col = f"mean_delta_{metric_col}"

        # 1) build hue-aware pairs in a deterministic order
        pairs = build_pairs_by_dataset(dataset_order, comparisons)

        # 2) build custom_texts in the same order as `pairs`
        custom_texts = []
        for (ds, ma), (_, mb) in pairs:
            # find matching row for this dataset and unordered model pair
            sub = df_model_comp[df_model_comp["datasets"] == ds]

            row = sub[
                ((sub["model_a"] == ma) & (sub["model_b"] == mb))
                | ((sub["model_a"] == mb) & (sub["model_b"] == ma))
            ]

            if row.shape[0] == 0:
                custom_texts.append("")  # no annotation available
                continue

            r = row.iloc[0]
            custom_texts.append(
                _format_comparison_text(
                    r,
                    delta_col=delta_col,
                    annot_style=annot_style,
                    use_q=True,
                    just_significance=annot_just_significance,
                    no_sig_name=annot_no_sig_name,
                )
            )

        # 3) draw
        annotator = Annotator(
            ax_main,
            pairs,
            data=df,
            x="datasets",
            y=metric_col,
            hue="model_type",
            order=dataset_order,
            hue_order=model_order,
        )
        annotator.configure(
            test="Wilcoxon",
            text_format="simple",
            comparisons_correction=None,
            loc=annotator_loc,
            line_offset=annotator_line_offset,
            text_offset=annotator_text_offset,
        )
        annotator.set_custom_annotations(custom_texts)
        annotator.annotate()

    # layout
    if legend_outside and draw_legend:
        fig.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        fig.tight_layout()

    return fig, (ax_main, ax_matrix), annotator


def save_legend_only(
    handles, labels, outpath, title="Model", ncol=3, figsize=(5.5, 0.8), dpi=600
):
    fig_leg, ax_leg = plt.subplots(figsize=figsize)
    ax_leg.axis("off")
    ax_leg.legend(
        handles=handles,
        labels=labels,
        title=title,
        ncol=ncol,
        loc="center",
        frameon=True,
        columnspacing=1.5,
        handletextpad=0.6,
    )
    fig_leg.savefig(outpath, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig_leg)


def plot_model_comparison_with_features(
    df,
    eval_set,
    feature_matrix,
    color_map=None,  # typically dataset -> color (for boxes)
    dataset_order=None,
    df_dset_comp=None,
    figsize_tuple=(6, 8),
    fig_ratio=(6, 6, 6, 3),
    shareY=True,
    title="Performance boxplots",
    random_performance_value=None,
    setMinYToRandomPerformance=False,
    # --- new toggles / styling ---
    show_boxplot=True,
    show_stripplot=True,
    show_medians=True,
    strip_size=5,
    strip_alpha=0.45,
    strip_face_alpha=0.6,
    strip_edge_alpha=1.0,
    strip_jitter=0.25,
    median_marker="_",
    median_markersize=30,
    median_color="lightgray",
    median_linewidth=1.2,
    add_background=True,  # uses your add_background_bands
    add_vertical_guides=True,  # your old axvline behavior
    boxplot_kwargs=None,
    annotator_loc="inside",
    annotator_line_offset=0.02,
    annotator_text_offset=1,  # points
    annot_style="threshold_float",
    annot_just_significance=False,
    annot_no_sig_name=True,
    y_label=None,
):
    """
    4-panel plot: one axis per model_type (pnet/rf/lr) plus feature matrix.
    Used to make the P1000 dataset-dataset comparison plots, with one panel per model type.

    - If show_boxplot: uses your existing plot_boxplot per panel.
    - If show_stripplot: overlays (or replaces) a stripplot of points per panel.
    - If show_medians: overlays per-dataset medians as underscore markers.

    Expects df columns:
      - 'datasets', 'model_type', f'{eval_set}_avg_precision'
    """
    metric_col = f"{eval_set}_avg_precision"

    # Use your helper if available; otherwise keep your existing ordering logic.
    try:
        valid_dataset_order = make_dataset_order(
            df, dataset_order, group_col="datasets"
        )
    except NameError:
        datasets = df["datasets"].unique().tolist()
        valid_dataset_order = (
            get_valid_group_order(df, dataset_order, group_col="datasets")
            if dataset_order
            else datasets
        )

    # Split data
    df_pnet = df[df["model_type"] == "pnet"]
    df_rf = df[df["model_type"] == "rf"]
    df_lr = df[df["model_type"] == "logistic_regression"]

    # Set up figure
    fig, (ax_pnet, ax_rf, ax_lr, ax_matrix) = plt.subplots(
        4,
        1,
        figsize=figsize_tuple,
        gridspec_kw={
            "height_ratios": list(fig_ratio),
            # "hspace": 0.08,
        },
        sharex=True,
    )

    # Optional background bands (helps readability, also matches your newer plots)
    if add_background:
        try:
            add_background_bands(
                [ax_pnet, ax_rf, ax_lr, ax_matrix], valid_dataset_order
            )
        except NameError:
            pass

    def _plot_panel(ax, df_panel, panel_title, panel_y_label=None):
        # 1) Boxplot (optional)
        if show_boxplot:
            _boxplot_kwargs = dict(
                dodge=False, ax=ax, showfliers=False, boxprops={"alpha": 0.3}
            )
            if boxplot_kwargs:
                _boxplot_kwargs.update(boxplot_kwargs)
            sns.boxplot(
                data=df_panel,
                x="datasets",
                y=metric_col,
                order=valid_dataset_order,
                palette=color_map,
                **_boxplot_kwargs,
            )  # TODO: wip

            # plot_boxplot(
            #     data=df_panel,
            #     x_name="datasets",
            #     y_name=metric_col,
            #     color_map=color_map,
            #     ax=ax,
            #     dataset_order=valid_dataset_order,
            # )

        # 2) Stripplot (optional)
        # If boxplot is off, stripplot becomes the primary mark.
        if show_stripplot:
            # record collections so you can post-style if you want
            start_idx = len(ax.collections)
            sns.stripplot(
                data=df_panel,
                x="datasets",
                y=metric_col,
                hue="datasets",
                palette=color_map,
                hue_order=valid_dataset_order,
                order=valid_dataset_order,
                ax=ax,
                jitter=strip_jitter,
                size=strip_size,
                alpha=strip_alpha,
                zorder=3,
            )

            # remove the hue legend (otherwise it’s huge)
            ax.legend([], [], frameon=False)

            # If you want your nicer face/edge alpha styling, reuse your util:
            with contextlib.suppress(NameError):
                set_stripplot_face_edge_alpha(
                    ax,
                    face_alpha=strip_face_alpha,
                    edge_alpha=strip_edge_alpha,
                    start_collection_idx=start_idx,
                )

        # 3) Medians (optional) — robust + consistent across panels
        if show_medians:
            sns.pointplot(
                data=df_panel,
                x="datasets",
                y=metric_col,
                order=valid_dataset_order,
                hue="datasets",
                hue_order=valid_dataset_order,
                palette=color_map,  # dataset -> color
                estimator=np.median,
                errorbar=None,
                linestyle="none",
                dodge=False,
                markers="_",
                markersize=median_markersize,
                ax=ax,
                zorder=2,
            )

        ax.set_title(panel_title)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        if panel_y_label is not None:
            ax.set_ylabel(panel_y_label)

        # Random baseline
        if random_performance_value is not None:
            ax.axhline(
                y=random_performance_value,
                linestyle="--",
                color="black",
                linewidth=1,
                zorder=1,
            )

    _plot_panel(ax_pnet, df_pnet, "model = P-NET", panel_y_label=y_label)
    _plot_panel(ax_rf, df_rf, "model = Random Forest", panel_y_label=y_label)
    _plot_panel(ax_lr, df_lr, "model = Logistic Regression", panel_y_label=y_label)

    # Share y-lims across model panels
    if shareY:
        y_min = min(ax_pnet.get_ylim()[0], ax_rf.get_ylim()[0], ax_lr.get_ylim()[0])
        y_max = max(ax_pnet.get_ylim()[1], ax_rf.get_ylim()[1], ax_lr.get_ylim()[1])
        if setMinYToRandomPerformance and random_performance_value is not None:
            y_min = min(y_min, random_performance_value - 0.03)

        for ax in (ax_pnet, ax_rf, ax_lr):
            ax.set_ylim(y_min, y_max)

    # Vertical guides (your original behavior)
    if add_vertical_guides:
        for i in range(len(valid_dataset_order)):
            for ax in (ax_pnet, ax_rf, ax_lr, ax_matrix):
                ax.axvline(
                    x=i,
                    color="lightgray",
                    linestyle="--",
                    linewidth=1,
                    zorder=0,
                )

    # Feature matrix
    plot_feature_matrix(feature_matrix, x_labels=valid_dataset_order, ax=ax_matrix)

    # Add stat significance annotations (optional)
    if df_dset_comp is not None:
        if Annotator is None:
            raise ImportError(
                "statannotations is not installed/available, but df_dset_comp was provided."
            )

        # Keep only rows where both datasets exist in valid_dataset_order
        present_in_order_mask = df_dset_comp["dataset_a"].isin(
            valid_dataset_order
        ) & df_dset_comp["dataset_b"].isin(valid_dataset_order)
        df_all_pairs = df_dset_comp.loc[present_in_order_mask].copy()
        if df_all_pairs.shape[0] > 0:
            # Add ordering keys so pairs are sorted left->right then top->bottom
            order_map = {d: i for i, d in enumerate(valid_dataset_order)}
            df_all_pairs["_a_idx"] = df_all_pairs["dataset_a"].map(order_map)
            df_all_pairs["_b_idx"] = df_all_pairs["dataset_b"].map(order_map)
            df_all_pairs = df_all_pairs.sort_values(["_a_idx", "_b_idx"]).reset_index(
                drop=True
            )

            # For each plotting panel (model), gather the pairs/texts relevant to that panel and draw once
            panels = [
                ("pnet", ax_pnet, df_pnet),
                ("rf", ax_rf, df_rf),
                ("logistic_regression", ax_lr, df_lr),
            ]

            for model_type, ax, df_panel in panels:
                # Filter rows for this model_type if the column exists; otherwise use all rows
                df_comp_panel = df_all_pairs[
                    df_all_pairs["model_type"] == model_type
                ].copy()

                # Keep only pairs where both datasets are present in this panel's data
                present = set(df_panel["datasets"].unique())
                df_comp_panel = df_comp_panel[
                    df_comp_panel["dataset_a"].isin(present)
                    & df_comp_panel["dataset_b"].isin(present)
                ].copy()

                # Build ordered pairs and matching custom texts (same order as df_comp_panel)
                pairs = list(
                    zip(
                        df_comp_panel["dataset_a"].tolist(),
                        df_comp_panel["dataset_b"].tolist(),
                    )
                )
                custom_texts = [
                    _format_comparison_text(
                        row,
                        delta_col=f"mean_delta_{metric_col}",
                        annot_style=annot_style,
                        use_q=True,
                        just_significance=annot_just_significance,
                        no_sig_name=annot_no_sig_name,
                    )
                    for _, row in df_comp_panel.iterrows()
                ]

                # Create one Annotator for this axis/panel
                annotator = Annotator(
                    ax,
                    pairs,
                    data=df_panel,
                    x="datasets",
                    y=metric_col,
                    order=valid_dataset_order,
                )
                annotator.configure(
                    test="Wilcoxon",  # we already computed/loaded stats; re-running to compare as sanity check
                    text_format="simple",
                    comparisons_correction=None,
                    loc=annotator_loc,
                    line_offset=annotator_line_offset,
                    text_offset=annotator_text_offset,
                )
                annotator.set_custom_annotations(custom_texts)
                annotator.annotate()

    # cosmetics
    for ax in (ax_pnet, ax_rf, ax_lr):
        yticks = ax.get_yticks()
        fixed_yticks = yticks[yticks <= 1.0]
        ax.set_yticks(fixed_yticks)

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_paired_seeds(df_runs, a, b, labels=None, replicate_col="random_seed"):
    sub = df_runs[df_runs["datasets"].isin([a, b])]
    wide = sub.pivot(index=replicate_col, columns="datasets", values="rank")

    fig, ax = plt.subplots(figsize=(3.5, 4))
    for _, row in wide.iterrows():
        ax.plot([0, 1], [row[a], row[b]], marker="o", alpha=0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels if labels else [a, b], rotation=45, ha="right")
    ax.set_ylabel("BRCA2 rank (lower = better)")
    ax.set_title("Per-seed BRCA2 rank")

    plt.tight_layout()
    plt.show()


def get_feature_matrix(datasets, features=["somatic", "rare", "lof", "missense"]):
    "Build upset-style indicator matrix to accompany box plots"
    logger.debug("features:", features)
    for dataset in datasets:
        logger.debug(
            f"working on {dataset}: {[int(feature in dataset) for feature in features]}"
        )

    indicator_df = pd.DataFrame(
        {
            feature: [int(feature in dataset) for dataset in datasets]
            for feature in features
        },
        index=datasets,
    )

    return indicator_df


def get_feature_matrix_hierarchical(
    datasets,
    somatic_keyword="somatic",
    rarity_keywords=("rare",),  # could add back "common" if needed
    consequence_keywords=("lof", "missense"),
):
    """
    Build an upset-style indicator matrix. Used in place of the original `get_feature_matrix` function because nicer way of building hierarchical labels.

    - 'somatic' is a simple substring check.
    - Germline features are interaction indicators: rarity x consequence,
      computed as AND(substring(rarity), substring(consequence)).

    Example:
      "germline_rare_lof_missense" -> rare_lof=1 and rare_missense=1
      "germline_rare_missense"     -> rare_missense=1, rare_lof=0
      "somatic_amp ... rare_lof"   -> somatic=1, rare_lof=1
    """
    rows = []
    for ds in datasets:
        row = {}
        row["somatic"] = int(somatic_keyword in ds)

        for r in rarity_keywords:
            has_r = r in ds
            for c in consequence_keywords:
                row[f"{r} {c}"] = int(has_r and (c in ds))

        rows.append(row)
        logger.debug(f"Computed feature matrix indicators for {ds}: {row}")

    return pd.DataFrame(rows, index=list(datasets))


def plot_paired_gene_rank_with_feature_caption(
    df_gene_rank_runs,
    a,
    b,
    features,
    dataset_labels=None,
    title="Per-seed BRCA2 rank",
    replicate_col="random_seed",
):
    """
    Plot per-seed paired BRCA2 ranks for datasets a vs b, with an UpSet-style
    feature matrix caption underneath.
    """
    if dataset_labels is None:
        dataset_labels = [a, b]

    # --- Build feature matrix for caption ---
    feature_matrix = get_feature_matrix_hierarchical([a, b])

    # --- Prepare per-seed paired data ---
    sub = df_gene_rank_runs[df_gene_rank_runs["datasets"].isin([a, b])]
    wide = sub.pivot(index=replicate_col, columns="datasets", values="rank")

    # --- Figure layout ---
    fig = plt.figure(figsize=(4, 5))
    gs = GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[4, 0.6],
        hspace=0.05,
    )

    ax_main = fig.add_subplot(gs[0])
    ax_feat = fig.add_subplot(gs[1])

    # --- Main paired-seed plot ---
    for _, row in wide.iterrows():
        ax_main.plot([0, 1], [row[a], row[b]], marker="o", alpha=0.6)

    ax_main.set_xticks([0, 1])
    ax_main.set_xticklabels(["", ""])  # x labels handled by feature matrix
    ax_main.set_ylabel("BRCA2 rank (lower = better)")
    ax_main.set_title(title)

    # --- Feature matrix caption ---
    plot_feature_matrix(
        feature_matrix=feature_matrix,
        x_labels=[a, b],
        ax=ax_feat,
    )

    # IMPORTANT: align feature matrix x-axis with main plot
    ax_feat.set_xlim(ax_main.get_xlim())

    # polish caption axis to feel like an annotation
    for spine in ax_feat.spines.values():
        spine.set_visible(False)
    ax_feat.tick_params(left=False)

    # plt.tight_layout()
    # plt.show()
    return fig


############################################################
# Functions for P1000 BRCA2 plots
############################################################
def _make_df_vs_baseline(df_brca2_results, baseline, metric_col="mean_delta_rank"):
    """
    Filter comparisons that involve baseline and orient sign so that
    negative values mean 'better than baseline' (lower rank).
    Returns a df with columns:
      - comparison (the non-baseline dataset)
      - metric_col, ci_low, ci_high, dataset_a, dataset_b
    """
    df = df_brca2_results[
        (df_brca2_results["dataset_a"] == baseline)
        | (df_brca2_results["dataset_b"] == baseline)
    ].copy()

    # orient: metric is A - B. We want (other - baseline).
    # If dataset_a == baseline, flip sign so it's (dataset_b - baseline)
    flip = df["dataset_a"] == baseline
    df.loc[flip, metric_col] *= -1

    # also flip CI bounds correspondingly by swapping/negating
    # If CI is for mean_delta already, flipping sign should just negate both bounds
    df.loc[flip, ["ci_low", "ci_high"]] *= -1
    # After negation, low/high might swap
    low = df["ci_low"].copy()
    high = df["ci_high"].copy()
    df["ci_low"] = np.minimum(low, high)
    df["ci_high"] = np.maximum(low, high)

    # label = the non-baseline dataset
    df["comparison"] = np.where(
        df["dataset_a"] == baseline, df["dataset_b"], df["dataset_a"]
    )
    return df


def _plot_feature_glyph(
    ax,
    feature_matrix,
    baseline,
    comp,
    features,
    marker_size=70,
    glyph_col_spacing=0.75,
):
    """
    Draw a compact horizontal glyph:
      - baseline row: filled circles at y=0
      - comparator row: hollow circles at y=0.55
    Marker presence is encoded by filled (1) vs light/empty (0).
    """
    fm = feature_matrix.loc[[baseline, comp], features]

    base = fm.loc[baseline].astype(int).to_numpy()
    other = fm.loc[comp].astype(int).to_numpy()

    # ---- pack columns closer together using spacing factor ----
    x = np.arange(len(features)) * glyph_col_spacing

    # y positions (tight)
    y_base = np.zeros_like(x, dtype=float)
    y_comp = np.full_like(x, 0.55, dtype=float)  # 0.55 keeps them close

    base_colors = ["grey" if v == 1 else "white" for v in base]
    other_face = ["black" if v == 1 else "white" for v in other]

    ax.scatter(
        x,
        y_base,
        s=marker_size,
        c=base_colors,
        edgecolor="grey",
        linewidth=1,
        marker="o",
    )

    ax.scatter(
        x,
        y_comp,
        s=marker_size,
        c=other_face,
        edgecolor="black",
        linewidth=1,
        marker="o",
    )

    # set x-limits consistent with compressed x
    left = -0.5 * glyph_col_spacing
    right = (len(features) - 1) * glyph_col_spacing + 0.5 * glyph_col_spacing
    ax.set_xlim(left, right)

    # tighten y-limits so circles nearly touch
    ax.set_ylim(-0.2, 0.9)

    ax.axis("off")


def plot_forest_vs_baseline_with_feature_glyphs(
    df_brca2_results,
    baseline,
    features,
    get_feature_matrix_func,
    title,
    metric_col="mean_delta_rank",
    q_col="q_fdr_bh",
    show_stars=True,
    figsize=None,
    right_pad=0.76,  # room for glyphs + header
    glyph_w_in=1.2,  # glyph inset width (inches)
    glyph_h_in=0.33,  # glyph inset height (inches)
    glyph_marker_size=55,
    glyph_x=-0.4,  # x position of glyph column (axes fraction)
    glyph_col_spacing=0.75,
    header_pad_y=0.03,  # header offset (axes fraction)
    legend_anchor=(0.99, 0.02),  # legend position in figure fraction
):
    """
    Forest plot of (comparison - baseline) with per-row horizontal feature glyphs.
    Forest plot of BRCA2 rank changes relative to a baseline, with per-row
    horizontal feature glyphs

    Each row shows:
      - point estimate and CI of (comparison - baseline)
      - a horizontal glyph encoding feature inclusion for baseline (bottom row)
        and comparator (top row)

    Parameters
    ----------
    df_brca2_results : pd.DataFrame
        Output table containing dataset_a/dataset_b comparisons.
    baseline : str
        Dataset name to use as baseline.
    features : list[str]
        Ordered list of feature categories used by get_feature_matrix_func.
    get_feature_matrix_func : callable
        Function like plot_utils.get_feature_matrix_hierarchical(datasets) -> DataFrame indexed by dataset.
    title : str
        Plot title.
    metric_col : str
        Column containing mean delta metric.
    right_pad : float
        Fraction of figure width to allocate to the main forest axis; smaller -> more room for glyphs.
    glyph_width, glyph_height : str
        Inset size (axes-relative percentage strings).
    """
    df = _make_df_vs_baseline(
        df_brca2_results, baseline=baseline, metric_col=metric_col
    )
    df = df.sort_values(metric_col).reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise ValueError(f"No comparisons found involving baseline '{baseline}'")

    if figsize is None:
        figsize = (4.0, 0.45 * n + 2.0)

    y = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(
        left=0.15,  # left margin reserved for glyphs
        right=0.95,  # end of forest
        top=0.90,
    )
    # Forest plot
    ax.errorbar(
        df[metric_col],
        y,
        xerr=[df[metric_col] - df["ci_low"], df["ci_high"] - df[metric_col]],
        fmt="o",
    )
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_yticks(y)  # keep row positions
    ax.set_yticklabels([])  # remove labels
    ax.tick_params(axis="y", left=False)  # remove tick marks

    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("Δ BRCA2 rank (comparison - baseline)\n(negative = better)")
    ax.set_title(title, y=1.06)

    # --- alternating background bands to improve row readability ---
    # subtle two-tone rows (drawn behind points/lines)
    band_color = "whitesmoke"  # light gray
    band_alpha = 0.7
    for i_row in range(n):
        if i_row % 2 == 0:
            ymin = i_row - 0.5
            ymax = i_row + 0.5
            ax.axhspan(ymin, ymax, facecolor=band_color, alpha=band_alpha, zorder=0)
    # ensure the plot elements remain above the bands
    ax.set_zorder(10)

    # Stars
    if show_stars:
        sig_vals = df[q_col] if (q_col in df.columns) else df.get("p_wilcoxon", None)
        if sig_vals is not None:
            span = ax.get_xlim()[1] - ax.get_xlim()[0]
            for i, v in enumerate(sig_vals.to_numpy()):
                if np.isfinite(v) and (v < 0.05):
                    ax.text(
                        df["ci_high"].iloc[i] + 0.02 * span,
                        i,
                        "*",
                        va="center",
                        fontsize=12,
                    )

    # Feature matrix for all needed datasets
    all_datasets = [baseline] + df["comparison"].tolist()
    feature_matrix_all = get_feature_matrix_func(all_datasets)

    # --- Header axis (feature labels) above glyph column ---
    # Place a single header glyph at the top, aligned with the glyph column.
    # --- Header axis (feature labels) above glyph column ---
    ax_header = inset_axes(
        ax,
        width=glyph_w_in,
        height=0.37,
        loc="lower left",
        bbox_to_anchor=(glyph_x, -0.01),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    # compute same left/right as glyphs (must match _plot_feature_glyph spacing)
    left = -0.5 * glyph_col_spacing
    right = (len(features) - 1) * glyph_col_spacing + 0.5 * glyph_col_spacing
    ax_header.set_xlim(left, right)
    ax_header.set_ylim(0, 1)

    ax_header.set_xticks(np.arange(len(features)) * glyph_col_spacing)
    ax_header.set_xticklabels(features, rotation=90, fontsize=10)

    ax_header.xaxis.set_ticks_position("bottom")
    ax_header.tick_params(
        axis="x",
        top=False,
        labeltop=False,
        bottom=True,
        labelbottom=True,
        pad=2,
        length=0,
        width=0,
    )
    ax_header.set_yticks([])
    for spine in ax_header.spines.values():
        spine.set_visible(False)
    for lab in ax_header.get_xticklabels():
        lab.set_clip_on(False)

    # --- tiny glyph color legend (comparison vs baseline) ---
    ax_key = inset_axes(
        ax,
        width=glyph_w_in,
        height=0.33,
        loc="lower left",  # anchor from bottom of inset
        bbox_to_anchor=(glyph_x, 0.97),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    ax_key.axis("off")

    # comparison (black) — TOP
    ax_key.scatter(
        [0.1],
        [0.80],
        s=70,
        c="black",
        edgecolor="black",
    )
    ax_key.text(0.2, 0.70, "comparison", va="center", fontsize=9)

    # baseline (grey) — BOTTOM
    ax_key.scatter(
        [0.1],
        [0.30],
        s=70,
        c="0.6",
        edgecolor="0.6",
    )
    ax_key.text(0.2, 0.30, "baseline", va="center", fontsize=9)

    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)

    # --- Per-row glyphs ---
    for i, comp in enumerate(df["comparison"].tolist()):
        y_frac = (i + 0.5) / n

        ax_in = inset_axes(
            ax,
            width=glyph_w_in,
            height=glyph_h_in,
            loc="center left",
            bbox_to_anchor=(glyph_x, y_frac),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        _plot_feature_glyph(
            ax=ax_in,
            feature_matrix=feature_matrix_all,
            baseline=baseline,
            comp=comp,
            features=features,
            marker_size=glyph_marker_size,
            glyph_col_spacing=glyph_col_spacing,
        )

    # plt.tight_layout()
    return fig, ax


############################################################
# Functions for 1D, single-gene spike-in simulations
############################################################
