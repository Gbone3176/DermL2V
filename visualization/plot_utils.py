import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN


def draw_kde_envelopes(
    ax,
    plot_df,
    palette,
    x_col="x",
    y_col="y",
    label_col="label",
    enabled=True,
    alpha=0.18,
    levels=1,
    thresh=0.25,
    bw_adjust=0.9,
    min_points=25,
    cluster_eps=None,
    cluster_min_samples=8,
):
    if not enabled or plot_df.empty:
        return

    contour_levels = max(int(levels), 2)

    for label, group in plot_df.groupby(label_col):
        if len(group) < min_points:
            continue

        coords = group[[x_col, y_col]].to_numpy(dtype=np.float32, copy=False)
        if coords.shape[0] < max(min_points, cluster_min_samples):
            continue

        eps = cluster_eps
        if eps is None:
            span = np.ptp(coords, axis=0)
            diag = float(np.linalg.norm(span))
            eps = max(diag * 0.08, 1e-3)

        clustering = DBSCAN(eps=eps, min_samples=cluster_min_samples)
        cluster_ids = clustering.fit_predict(coords)

        for cluster_id in np.unique(cluster_ids):
            if cluster_id < 0:
                continue

            cluster_group = group.iloc[cluster_ids == cluster_id]
            if len(cluster_group) < min_points:
                continue

            sns.kdeplot(
                data=cluster_group,
                x=x_col,
                y=y_col,
                ax=ax,
                fill=True,
                levels=contour_levels,
                thresh=thresh,
                bw_adjust=bw_adjust,
                alpha=alpha,
                color=palette.get(label),
                warn_singular=False,
            )
