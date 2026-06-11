#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import importlib
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate_crown import evaluate_crowns


# ---- Load config ----
preparser = sys.argv
config_name = next((preparser[i + 1] for i, v in enumerate(preparser)
                    if v == "--config" and i + 1 < len(preparser)), "config")
config = importlib.import_module(config_name)


def plot_crown_summary(df, out_dir):
    sns.set(style="ticks", context="talk")
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.scatterplot(
        x="GT_Diameter_m", y="Pred_Diameter_m",
        data=df, hue="max_conf", palette="viridis", s=40,
        alpha=0.7, edgecolor=None, ax=ax
    )

    max_d = max(df["GT_Diameter_m"].max(), df["Pred_Diameter_m"].max())
    ax.plot([0, max_d], [0, max_d], "r--", lw=2)
    m, b = np.polyfit(df["GT_Diameter_m"], df["Pred_Diameter_m"], 1)
    r2 = np.corrcoef(df["GT_Diameter_m"], df["Pred_Diameter_m"])[0, 1] ** 2
    rmse = np.sqrt(np.mean((df["Pred_Diameter_m"] - df["GT_Diameter_m"]) ** 2))
    bias = np.mean((df["Pred_Diameter_m"] - df["GT_Diameter_m"]) / df["GT_Diameter_m"]) * 100

    ax.text(0.05, 0.95,
            f"$y={m:.2f}x+{b:.2f}$\n$R^2$={r2:.2f}\nRMSE={rmse:.2f} m\nBias={bias:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=12,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel("Labelled Crown Diameter (m)", fontsize=14)
    ax.set_ylabel("Predicted Crown Diameter (m)", fontsize=14)
    ax.set_aspect("equal", "box")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "crown_eval_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved plot: {out_path}")


def main():
    matched, precision, recall, f1, rmse, bias, stats = evaluate_crowns(
        pred_point_path=config.pred_point_path,
        pred_polygon_path=config.pred_polygon_path,
        gt_polygon_path=config.gt_polygon_path,
        iou_thresh=getattr(config, "iou_thresh", 0.3)
    )

    print("\n===== Crown Evaluation Results =====")
    print(f"Pred crowns: {stats['n_pred']}")
    print(f"GT crowns:   {stats['n_gt']}")
    print(f"Matched:     {stats['n_matched']}")
    print(f"Precision:   {precision:.3f}")
    print(f"Recall:      {recall:.3f}")
    print(f"F1 Score:    {f1:.3f}")
    print(f"RMSE (m):    {rmse:.3f}")
    print(f"Bias (%):    {bias:.2f}")
    print("===================================\n")

    if not matched:
        print("⚠️ No matches found. No outputs exported.")
        return

    # ---- Build DataFrame ----
    records = []
    for pred_area, gt_area, pred_geom, gt_geom, iou, avg_conf, max_conf in matched:
        pred_diam = 2 * np.sqrt(pred_area / np.pi)
        gt_diam = 2 * np.sqrt(gt_area / np.pi)
        abs_err = pred_diam - gt_diam
        rel_err = abs_err / gt_diam
        records.append({
            "GT_Diameter_m": gt_diam,
            "Pred_Diameter_m": pred_diam,
            "Abs_Error_m": abs_err,
            "Rel_Error": rel_err,
            "IoU": iou,
            "avg_conf": avg_conf,
            "max_conf": max_conf
        })

    df = pd.DataFrame(records)
    plot_crown_summary(df, config._directory)

    # ---- Exports ----
    df.to_csv(os.path.join(config.log_directory, "matched_crowns.csv"), index=False)
    print(f"✅ Saved matched crowns table → matched_crowns.csv")

    # gdf_pred = gpd.GeoDataFrame(df.copy(),
    #                             geometry=[m[2] for m in matched],
    #                             crs=gpd.read_file(config.pred_polygon_path).crs)
    # gdf_gt = gpd.GeoDataFrame(df.copy(),
    #                           geometry=[m[3] for m in matched],
    #                           crs=gpd.read_file(config.gt_polygon_path).crs)

    # gpkg_path = os.path.join(config.log_directory, "matched_crowns.gpkg")
    # gdf_gt.to_file(gpkg_path, layer="ground_truth", driver="GPKG")
    # gdf_pred.to_file(gpkg_path, layer="predicted", driver="GPKG")
    # print(f"✅ Saved matched polygons → {gpkg_path}")

    # ---- Summary CSV ----
    summary_path = os.path.join(config.log_directory, "crown_eval_summary.csv")
    pd.DataFrame([{
        "n_pred": stats["n_pred"],
        "n_gt": stats["n_gt"],
        "n_matched": stats["n_matched"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rmse_m": rmse,
        "bias_percent": bias
    }]).to_csv(summary_path, index=False)
    print(f"✅ Saved summary metrics → {summary_path}")


if __name__ == "__main__":
    main()
