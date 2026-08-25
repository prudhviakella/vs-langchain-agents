"""
MLflow Capabilities Demo
------------------------
A simple, self-contained script that demonstrates the most commonly used
MLflow tracking APIs using a fake "model training" scenario.

No real model training happens here — we simulate metric values so you can
focus entirely on understanding what each MLflow API does and where the
results appear in the UI.

What this script covers
-----------------------
  1. mlflow.set_tracking_uri()     — point MLflow at your server
  2. mlflow.set_experiment()       — organise runs into named experiments
  3. mlflow.start_run()            — open a run (the container for everything)
  4. mlflow.log_param()            — log a single config value
  5. mlflow.log_params()           — log many config values at once
  6. mlflow.log_metric()           — log a single numeric value
  7. mlflow.log_metrics()          — log many numeric values at once
  8. mlflow.log_metric(step=i)     — log a metric over time (shows as a chart)
  9. mlflow.set_tag()              — attach a label/note to the run
  10. mlflow.log_artifact()        — upload any file (CSV, JSON, plot, etc.)
  11. mlflow.log_text()            — log a string directly as a file artifact
  12. mlflow.log_dict()            — log a dict directly as a JSON artifact
  13. mlflow.log_figure()          — log a matplotlib figure as an artifact
  14. mlflow.get_artifact_uri()    — print where artifacts are stored
  15. mlflow.MlflowClient          — query run data back programmatically

Prerequisites
-------------
  pip install mlflow matplotlib

Start the MLflow UI before running:
  mlflow ui --port 5001
  then open http://localhost:5001

Run:
  python mlflow_demo.py
"""

import json
import math
import random
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# ── Optional: matplotlib for the figure-logging demo ─────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TRACKING_URI    = "http://localhost:5001"
EXPERIMENT_NAME = "mlflow-capabilities-demo"
NUM_EPOCHS      = 20   # simulated training epochs for step-metric charts

random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — fake metric generators
# These simulate realistic training curves without any real model.
# ══════════════════════════════════════════════════════════════════════════════

def simulate_loss(epoch, total):
    """Exponential decay with small noise — looks like a real training loss curve."""
    base  = math.exp(-epoch / (total * 0.4))
    noise = random.uniform(-0.02, 0.02)
    return round(max(0.01, base + noise), 4)


def simulate_accuracy(epoch, total):
    """Sigmoid ramp-up with small noise — looks like a real accuracy curve."""
    base  = 1 / (1 + math.exp(-10 * (epoch / total - 0.5)))
    noise = random.uniform(-0.01, 0.01)
    return round(min(1.0, max(0.0, base + noise)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Connect to MLflow server and create / select an experiment
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  MLflow Capabilities Demo")
print("═" * 60)

# 1. set_tracking_uri
#    Tells MLflow where to send all data.
#    Without this it defaults to a local ./mlruns folder.
mlflow.set_tracking_uri(TRACKING_URI)
print(f"\n[1] Tracking URI set → {TRACKING_URI}")

# 2. set_experiment
#    All runs below will be grouped under this experiment name.
#    Creates the experiment automatically if it does not exist.
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"[2] Experiment     → '{EXPERIMENT_NAME}'")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Open a run and log everything
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Starting MLflow run ──────────────────────────────────────")

with mlflow.start_run(run_name="demo-run") as run:

    run_id = run.info.run_id
    print(f"    Run ID: {run_id}")

    # ── 3. log_param / log_params ────────────────────────────────────────────
    #
    # Params are CONFIGURATION values — things you set before training starts.
    # They are strings in MLflow (numbers are auto-converted).
    # Visible in UI → Params column and the run detail page.
    # Use them to understand WHAT settings produced a given result.
    #
    # log_param()  → one key-value pair at a time
    # log_params() → dict of many key-value pairs in one call (preferred)

    mlflow.log_param("learning_rate", 0.001)          # single param
    print("\n[3] log_param()  → learning_rate = 0.001")

    mlflow.log_params({                                # batch of params
        "model_type":   "transformer",
        "batch_size":   32,
        "num_epochs":   NUM_EPOCHS,
        "optimizer":    "adam",
        "dropout":      0.1,
        "dataset":      "hotpotqa",
    })
    print("[4] log_params() → 6 params logged at once")

    # ── 4. set_tag ───────────────────────────────────────────────────────────
    #
    # Tags are free-form labels. Unlike params they CAN be updated after the
    # run ends. Great for notes, environment info, or experiment annotations.
    # Visible in UI → Tags column.

    mlflow.set_tag("author",      "prudhvi")
    mlflow.set_tag("environment", "local-dev")
    mlflow.set_tag("notes",       "demo run — no real model trained")
    print("\n[5] set_tag()    → 3 tags attached to the run")
    #
    # ── 5. log_metric / log_metrics ──────────────────────────────────────────
    #
    # Metrics are NUMERIC RESULTS — things you measure during or after training.
    # log_metric()  → one metric at a time
    # log_metrics() → dict of many metrics in one call (preferred for final scores)
    #
    # Without a step argument these are "scalar" metrics — a single value.
    # Visible in UI → Metrics column.

    mlflow.log_metric("final_accuracy", 0.923)
    print("\n[6] log_metric()  → final_accuracy = 0.923")

    mlflow.log_metrics({
        "final_loss":      0.082,
        "val_accuracy":    0.911,
        "val_loss":        0.094,
        "f1_score":        0.918,
        "precision":       0.925,
        "recall":          0.912,
    })
    print("[7] log_metrics() → 6 final metrics logged at once")
    #
    # ── 6. log_metric with step — time-series charts ──────────────────────────
    #
    # When you pass step=i, MLflow records the metric value AT that step.
    # This creates a LINE CHART in the UI — perfect for loss/accuracy curves.
    # Visible in UI → run detail page → Metrics tab → click a metric name.
    #
    # Rule of thumb:
    #   No step  → final summary scalar (one number)
    #   With step → curve over time (many numbers, one per step)

    print(f"\n[8] log_metric(step=i) → logging loss + accuracy for {NUM_EPOCHS} epochs")
    # for epoch in range(NUM_EPOCHS):
    #     train_loss = simulate_loss(epoch, NUM_EPOCHS)
    #     val_loss   = simulate_loss(epoch + 2, NUM_EPOCHS)   # val lags slightly
    #     train_acc  = simulate_accuracy(epoch, NUM_EPOCHS)
    #     val_acc    = simulate_accuracy(epoch - 1, NUM_EPOCHS)
    #
    #     # Log all four step-metrics together so they share the same step index
    #     mlflow.log_metrics({
    #         "train_loss":     train_loss,
    #         "val_loss":       val_loss,
    #         "train_accuracy": train_acc,
    #         "val_accuracy_curve": val_acc,
    #     }, step=epoch)
    #
    #     if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
    #         print(f"    epoch {epoch:>2}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
    for i in range(1,20):

        # Log all four step-metrics together so they share the same step index
        mlflow.log_metrics({
            "first_metric": 10 + i,
            "second_metric": 20 + i,
            "third_metric": 30 + i,
            "fourth_metric": 40 + i,
        }, step=i)
    #
    # ── 7. log_artifact — upload any file ─────────────────────────────────────
    #
    # Artifacts are FILES attached to the run — anything you want to save:
    # model weights, evaluation results, plots, configs, reports, etc.
    # Visible in UI → run detail page → Artifacts tab.
    #
    # You pass a local file path; MLflow copies it to the artifact store.

    # Create a sample CSV results file and log it
    results_csv_path = Path(tempfile.gettempdir()) / "eval_results.csv"
    with results_csv_path.open("w") as f:
        f.write("epoch,train_loss,val_loss,train_accuracy\n")
        for epoch in range(NUM_EPOCHS):
            f.write(f"{epoch},{simulate_loss(epoch, NUM_EPOCHS)},"
                    f"{simulate_loss(epoch+2, NUM_EPOCHS)},"
                    f"{simulate_accuracy(epoch, NUM_EPOCHS)}\n")

    mlflow.log_artifact(str(results_csv_path))
    # print(f"\n[9] log_artifact() → {results_csv_path.name} uploaded to artifact store")
    #
    # ── 8. log_text — log a string directly as a file ─────────────────────────
    #
    # Convenience wrapper: no need to write a temp file yourself.
    # The string is saved as a named file in the artifact store.
    # Great for logging prompts, configs, or small text reports inline.

    summary_text = (
        "=== Run Summary ===\n"
        f"Model       : transformer\n"
        f"Dataset     : hotpotqa\n"
        f"Epochs      : {NUM_EPOCHS}\n"
        f"Final Acc   : 0.923\n"
        f"Final Loss  : 0.082\n"
        "Notes: demo run — values are simulated\n"
    )
    mlflow.log_text(summary_text, "run_summary.txt")
    print("[10] log_text()    → run_summary.txt saved as artifact")

    # ── 9. log_dict — log a Python dict directly as JSON ──────────────────────
    #
    # Saves a dict as a JSON file artifact in one line.
    # Perfect for logging hyperparameter configs, evaluation results, or metadata.

    config_dict = {
        "model": {
            "type":          "transformer",
            "hidden_size":   256,
            "num_layers":    4,
            "attention_heads": 8,
            "dropout":       0.1,
        },
        "training": {
            "optimizer":     "adam",
            "learning_rate": 0.001,
            "batch_size":    32,
            "epochs":        NUM_EPOCHS,
        },
        "data": {
            "dataset":       "hotpotqa",
            "train_split":   0.8,
            "val_split":     0.2,
        }
    }
    mlflow.log_dict(config_dict, "full_config.json")
    print("[11] log_dict()    → full_config.json saved as artifact")

    # ── 10. log_figure — log a matplotlib plot ────────────────────────────────
    #
    # Saves a matplotlib Figure object directly as an image artifact.
    # No need to save to disk first — MLflow handles the temp file internally.
    # Visible in UI → Artifacts tab → click the image file.

    if HAS_MATPLOTLIB:
        epochs     = list(range(NUM_EPOCHS))
        train_losses = [simulate_loss(e, NUM_EPOCHS) for e in epochs]
        val_losses   = [simulate_loss(e + 2, NUM_EPOCHS) for e in epochs]
        train_accs   = [simulate_accuracy(e, NUM_EPOCHS) for e in epochs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, train_losses, label="Train Loss", color="steelblue")
        ax1.plot(epochs, val_losses,   label="Val Loss",   color="coral", linestyle="--")
        ax1.set_title("Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_accs, label="Train Accuracy", color="seagreen")
        ax2.set_title("Accuracy Curve")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Training Curves (Simulated)", fontsize=13)
        plt.tight_layout()

        mlflow.log_figure(fig, "training_curves.png")
        plt.close(fig)
        print("[12] log_figure()  → training_curves.png saved as artifact")
    else:
        print("[12] log_figure()  → skipped (matplotlib not installed)")

    # ── 11. get_artifact_uri — where are my artifacts stored? ─────────────────
    #
    # Returns the root URI of the artifact store for this run.
    # Useful for debugging or constructing direct paths to uploaded files.

    artifact_uri = mlflow.get_artifact_uri()
    print(f"\n[13] get_artifact_uri() → {artifact_uri}")

    print("\n── Run complete ─────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Read run data back with MlflowClient
# ══════════════════════════════════════════════════════════════════════════════
#
# MlflowClient lets you QUERY MLflow programmatically — useful for
# comparing runs, building reports, or triggering downstream steps.

print("\n── Reading run data back with MlflowClient ──────────────────")

client    = MlflowClient(tracking_uri=TRACKING_URI)
run_data  = client.get_run(run_id)
print(run_data)

# Params — the config values we logged
print("\n[14] MlflowClient.get_run() → params")
for key, value in sorted(run_data.data.params.items()):
    print(f"    {key:<20} = {value}")

# Metrics — the final scalar values (not the step-series)
print("\n[15] MlflowClient.get_run() → scalar metrics")
for key, value in sorted(run_data.data.metrics.items()):
    print(f"    {key:<25} = {value:.4f}")

# Step-series history — list of (step, value) for a time-series metric
print("\n[16] MlflowClient.get_metric_history() → train_loss over epochs")
history = client.get_metric_history(run_id, "train_loss")
for entry in history[:5]:   # print first 5 steps
    print(f"    step={entry.step:>2}  train_loss={entry.value:.4f}")
print(f"    ... ({len(history)} steps total)")

# Tags
print("\n[17] MlflowClient.get_run() → tags")
for key, value in run_data.data.tags.items():
    if not key.startswith("mlflow."):   # skip MLflow's internal system tags
        print(f"    {key:<20} = {value}")
#
#
# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  All done! Open the MLflow UI to explore results:")
print(f"  → {TRACKING_URI}")
print()
print("  What to look at in the UI:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  Experiments list    → select 'mlflow-capabilities-  │")
print("  │                         demo'                        │")
print("  │  Runs table          → see params + final metrics    │")
print("  │  Run detail page:                                     │")
print("  │    Params tab        → all log_param / log_params    │")
print("  │    Metrics tab       → scalar + step-chart metrics   │")
print("  │    Tags tab          → all set_tag values            │")
print("  │    Artifacts tab     → CSV, TXT, JSON, PNG files     │")
print("  └──────────────────────────────────────────────────────┘")
print("═" * 60 + "\n")
