import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def generate_performance_charts(
    history_json_path: str = "d:\\Major Project\\src\\models\\checkpoints\\training_history.json",
    benchmark_csv_path: str = "d:\\Major Project\\docs\\benchmark_results.csv",
    output_dir: str = "d:\\Major Project\\docs\\figures"
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Generating report & presentation figures in: {output_dir}")
    curves_path = None

    # Set dark aesthetic style for charts matching project theme
    plt.style.use('dark_background')
    fig_bg = '#0a0e17'
    panel_bg = '#161c2d'

    # -------------------------------------------------------------
    # 1. Training Curves Plot (Loss & CER vs Epochs)
    # -------------------------------------------------------------
    # These lists used to be pre-filled with an invented convergence curve
    # (train_loss 2.45->0.54, val_cer 0.65->0.15) that was plotted whenever
    # training_history.json was missing or failed to parse. A figure captioned
    # "Supervised CTC Training & Convergence Curves" showing numbers no run
    # produced does not belong in a submitted report. There is now no fallback:
    # if the history is not there, the chart is not drawn.
    if not os.path.exists(history_json_path):
        print(f"[!] No training history at {history_json_path} - skipping the training-curve")
        print("    figure. Run 'python -m src.train' first; no placeholder will be drawn.")
        hist = None
    else:
        with open(history_json_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        if not hist:
            print(f"[!] {history_json_path} is empty - skipping the training-curve figure.")
            hist = None

    if hist:
        epochs = [h.get("epoch", idx + 1) for idx, h in enumerate(hist)]
        train_loss = [h.get("train_loss") for h in hist]
        val_loss = [h.get("val_loss") for h in hist]
        val_cer = [h.get("val_cer") for h in hist]
        if any(v is None for v in train_loss + val_loss + val_cer):
            raise ValueError(
                f"{history_json_path} has entries missing train_loss/val_loss/val_cer. "
                f"Refusing to plot partial data as if it were complete."
            )

        fig, ax1 = plt.subplots(figsize=(9, 5), facecolor=fig_bg)
        ax1.set_facecolor(panel_bg)

        color_train = '#00f2fe'
        color_val = '#4facfe'
        color_cer = '#7f00ff'

        ax1.plot(epochs, train_loss, 'o-', color=color_train, linewidth=2.5, label='Train Loss')
        ax1.plot(epochs, val_loss, 's--', color=color_val, linewidth=2.5, label='Validation Loss')
        ax1.set_xlabel('Epochs', fontsize=11, color='#f1f5f9', fontweight='bold')
        ax1.set_ylabel('CTC Loss', fontsize=11, color=color_train, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_train)
        ax1.grid(True, linestyle=':', alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(epochs, val_cer, '^:', color=color_cer, linewidth=2.5, label='Val CER')
        ax2.set_ylabel('Character Error Rate (CER)', fontsize=11, color=color_cer, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_cer)

        plt.title('Supervised CTC Training & Convergence Curves', fontsize=13, color='#ffffff', fontweight='bold', pad=15)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', facecolor='#0d1322', edgecolor='#333333')

        plt.tight_layout()
        curves_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(curves_path, dpi=300, bbox_inches='tight', facecolor=fig_bg)
        plt.close()
        print(f"  [+] Saved: '{curves_path}'  ({len(epochs)} epochs of real history)")

    # -------------------------------------------------------------
    # 2. Benchmark Comparison Bar Chart (CNN vs Transformer)
    # -------------------------------------------------------------
    # As above: params_m = [5.28, 62.0], latency_ms = [65.4, 380.2] and
    # cer_scores = [0.12, 0.08] used to be plotted whenever the benchmark CSV
    # was missing or any field failed to parse - and every `except: pass` below
    # meant a single bad field silently reverted that bar to the invented
    # number while its neighbours showed real data. A CER of 0.12/0.08 was never
    # measured by this project. No fallback now: no CSV, no chart.
    import csv

    if not os.path.exists(benchmark_csv_path):
        print(f"[!] No benchmark results at {benchmark_csv_path} - skipping the comparison")
        print("    figure. Run 'python -m src.evaluate' first.")
        return {"training_curves": curves_path, "benchmark_comparison": None}

    with open(benchmark_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    models, params_m, latency_ms, cer_scores = [], [], [], []
    for r in rows:
        name = r.get("Model Architecture", "").strip()
        if not name:
            continue
        try:
            params = float(r["Parameters Count"])
            latency = float(r["Avg Latency (ms)"])
            cer = float(r["CER"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Row {name!r} in {benchmark_csv_path} is missing or has a non-numeric "
                f"Parameters Count / Avg Latency (ms) / CER ({exc}). Refusing to "
                f"substitute a placeholder value into a report figure."
            ) from exc
        models.append(name.replace(" (", "\n("))
        params_m.append(round(params / 1e6, 2))
        latency_ms.append(round(latency, 1))
        cer_scores.append(round(cer, 3))

    if not models:
        print(f"[!] {benchmark_csv_path} has no data rows - skipping the comparison figure.")
        return {"training_curves": curves_path, "benchmark_comparison": None}

    fig, (ax_bar1, ax_bar2, ax_bar3) = plt.subplots(1, 3, figsize=(15, 5), facecolor=fig_bg)
    for ax in (ax_bar1, ax_bar2, ax_bar3):
        ax.set_facecolor(panel_bg)
        ax.tick_params(axis='x', labelsize=7)

    colors = ['#00f2fe', '#7f00ff', '#4facfe', '#f59e0b'][:len(models)]

    # Model Parameters Comparison
    bars1 = ax_bar1.bar(models, params_m, color=colors, width=0.45, edgecolor='#333333')
    ax_bar1.set_title('Model Parameters (Lower is Lighter)', fontsize=11, color='#ffffff', fontweight='bold')
    ax_bar1.set_ylabel('Parameters (Millions)', fontsize=10, color='#94a3b8')
    ax_bar1.grid(axis='y', linestyle=':', alpha=0.3)
    for bar in bars1:
        yval = bar.get_height()
        ax_bar1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval} M', ha='center', va='bottom', color='#ffffff', fontweight='bold')

    # Inference Latency Comparison
    bars2 = ax_bar2.bar(models, latency_ms, color=colors, width=0.45, edgecolor='#333333')
    ax_bar2.set_title('Inference Latency (Lower is Faster)', fontsize=11, color='#ffffff', fontweight='bold')
    ax_bar2.set_ylabel('Avg Latency (ms/image)', fontsize=10, color='#94a3b8')
    ax_bar2.grid(axis='y', linestyle=':', alpha=0.3)
    for bar in bars2:
        yval = bar.get_height()
        ax_bar2.text(bar.get_x() + bar.get_width()/2.0, yval + 5.0, f'{yval} ms', ha='center', va='bottom', color='#ffffff', fontweight='bold')

    # Character Error Rate. This was parsed from the CSV but never plotted, so
    # the "architectural comparison" figure showed size and speed and omitted
    # the only metric the project is actually about.
    bars3 = ax_bar3.bar(models, cer_scores, color=colors, width=0.45, edgecolor='#333333')
    ax_bar3.set_title('Character Error Rate (Lower is Better)', fontsize=11, color='#ffffff', fontweight='bold')
    ax_bar3.set_ylabel('CER on held-out test split', fontsize=10, color='#94a3b8')
    ax_bar3.grid(axis='y', linestyle=':', alpha=0.3)
    ax_bar3.set_ylim(0, max(1.05, max(cer_scores) * 1.2))
    # A CER of 1.0 means "as many errors as there are characters" - worth a line.
    ax_bar3.axhline(1.0, color='#ef4444', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_bar3.text(len(models) - 0.5, 1.01, 'CER = 1.0', color='#ef4444', fontsize=8, ha='right')
    for bar in bars3:
        yval = bar.get_height()
        ax_bar3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', color='#ffffff', fontweight='bold')

    plt.suptitle('Empirical Architectural Comparison: Primary CNN vs Baseline Vision Transformer', fontsize=13, color='#ffffff', fontweight='bold', y=1.02)
    plt.tight_layout()
    bench_path = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(bench_path, dpi=300, bbox_inches='tight', facecolor=fig_bg)
    plt.close()
    print(f"  [+] Saved: '{bench_path}'  ({len(models)} model(s) from real benchmark data)")

    print("[OK] Figures generated from measured data only.")
    return {"training_curves": curves_path, "benchmark_comparison": bench_path}


if __name__ == "__main__":
    generate_performance_charts()
