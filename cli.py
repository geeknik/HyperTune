"""HyperTune CLI - Command-line interface for LLM hyperparameter tuning."""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from hypertune.core import HyperTune

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

RESPONSE_PREVIEW_LENGTH = 500


def truncate_text(text, max_length=RESPONSE_PREVIEW_LENGTH):
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


def format_score_breakdown(result):
    lines = []
    lines.append(f"  Total: {result['total_score']:.3f}")
    lines.append(f"  ├─ Coherence:  {result['coherence_score']:.3f} (40%)")
    lines.append(f"  ├─ Relevance:  {result['relevance_score']:.3f} (40%)")
    lines.append(f"  ├─ Complexity: {result['complexity_score']:.3f} (20%)")
    quality = result.get("quality_penalty", 1.0)
    if quality < 1.0:
        lines.append(f"  └─ Quality:    {quality:.3f} (penalty applied)")
    return "\n".join(lines)


def format_hyperparameters(hyperparams):
    return " | ".join(f"{k}={v}" for k, v in hyperparams.items())


def extract_common_terms(results, top_n=10):
    all_text = " ".join([result["text"] for result in results])
    words = word_tokenize(all_text.lower())
    stop_words = set(stopwords.words("english"))
    words = [
        word
        for word in words
        if word.isalnum() and word not in stop_words and len(word) > 2
    ]
    word_freq = Counter(words)
    return word_freq.most_common(top_n)


def save_results_json(results, args, output_path):
    export_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "prompt": args.prompt,
            "provider": args.provider,
            "model": args.model,
            "iterations": args.iterations,
        },
        "results": results,
        "summary": {
            "best_score": results[0]["total_score"] if results else 0,
            "best_hyperparameters": results[0]["hyperparameters"] if results else {},
            "scores_distribution": {
                "min": min(r["total_score"] for r in results) if results else 0,
                "max": max(r["total_score"] for r in results) if results else 0,
                "mean": np.mean([r["total_score"] for r in results]) if results else 0,
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    return output_path


def create_analysis_dashboard(results, output_dir="."):
    if len(results) < 2:
        return None

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("HyperTune Analysis Dashboard", fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    scores = [r["total_score"] for r in results]
    colors = [
        "#2ecc71" if s > 0.7 else "#f39c12" if s > 0.4 else "#e74c3c" for s in scores
    ]
    ax1.hist(
        scores,
        bins=min(10, len(results)),
        color="#3498db",
        edgecolor="white",
        alpha=0.7,
    )
    ax1.axvline(
        np.mean(scores),
        color="#e74c3c",
        linestyle="--",
        label=f"Mean: {np.mean(scores):.2f}",
    )
    ax1.axvline(
        scores[0],
        color="#2ecc71",
        linestyle="-",
        linewidth=2,
        label=f"Best: {scores[0]:.2f}",
    )
    ax1.set_xlabel("Total Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Score Distribution")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 1)

    ax2 = fig.add_subplot(gs[0, 1:])
    top_n = min(5, len(results))
    response_labels = [f"#{i + 1}" for i in range(top_n)]
    coherence = [r["coherence_score"] * 0.4 for r in results[:top_n]]
    relevance = [r["relevance_score"] * 0.4 for r in results[:top_n]]
    complexity = [r["complexity_score"] * 0.2 for r in results[:top_n]]

    x = np.arange(top_n)
    width = 0.6
    ax2.bar(x, coherence, width, label="Coherence (40%)", color="#3498db")
    ax2.bar(
        x, relevance, width, bottom=coherence, label="Relevance (40%)", color="#2ecc71"
    )
    ax2.bar(
        x,
        complexity,
        width,
        bottom=[c + r for c, r in zip(coherence, relevance)],
        label="Complexity (20%)",
        color="#9b59b6",
    )

    for i, r in enumerate(results[:top_n]):
        quality = r.get("quality_penalty", 1.0)
        if quality < 1.0:
            ax2.annotate(
                f"Q:{quality:.1f}", (i, 0.02), ha="center", fontsize=7, color="#e74c3c"
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(response_labels)
    ax2.set_ylabel("Weighted Score Contribution")
    ax2.set_title("Score Breakdown - Top Responses")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, 1.1)

    available_params = [
        p
        for p in [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "top_k",
        ]
        if p in results[0]["hyperparameters"]
    ]

    if len(available_params) >= 2:
        ax3 = fig.add_subplot(gs[1, 0])
        df = pd.DataFrame(
            [{**r["hyperparameters"], "score": r["total_score"]} for r in results]
        )
        corr_data = df[available_params + ["score"]].corr()
        mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
        sns.heatmap(
            corr_data,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            ax=ax3,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8},
        )
        ax3.set_title("Parameter Correlations")

    ax4 = fig.add_subplot(gs[1, 1:])
    df = pd.DataFrame(
        [{**r["hyperparameters"], "total_score": r["total_score"]} for r in results]
    )

    if "temperature" in available_params:
        scatter = ax4.scatter(
            df["temperature"],
            df["total_score"],
            c=df["total_score"],
            cmap="RdYlGn",
            s=60,
            alpha=0.7,
            edgecolors="white",
        )
        z = np.polyfit(df["temperature"], df["total_score"], 1)
        p = np.poly1d(z)
        temp_range = np.linspace(df["temperature"].min(), df["temperature"].max(), 100)
        ax4.plot(
            temp_range,
            p(temp_range),
            "--",
            color="#e74c3c",
            alpha=0.8,
            label=f"Trend (r={df['temperature'].corr(df['total_score']):.2f})",
        )
        ax4.set_xlabel("Temperature")
        ax4.set_ylabel("Total Score")
        ax4.set_title("Temperature vs Score (with trend)")
        ax4.legend(fontsize=8)
        ax4.set_ylim(0, 1.1)
        plt.colorbar(scatter, ax=ax4, shrink=0.8)

    ax5 = fig.add_subplot(gs[2, :])
    best = results[0]
    worst = results[-1]

    metrics = ["Coherence", "Relevance", "Complexity", "Quality", "Total"]
    best_vals = [
        best["coherence_score"],
        best["relevance_score"],
        best["complexity_score"],
        best.get("quality_penalty", 1.0),
        best["total_score"],
    ]
    worst_vals = [
        worst["coherence_score"],
        worst["relevance_score"],
        worst["complexity_score"],
        worst.get("quality_penalty", 1.0),
        worst["total_score"],
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax5.bar(
        x - width / 2,
        best_vals,
        width,
        label=f"Best (#{1})",
        color="#2ecc71",
        alpha=0.8,
    )
    ax5.bar(
        x + width / 2,
        worst_vals,
        width,
        label=f"Worst (#{len(results)})",
        color="#e74c3c",
        alpha=0.8,
    )
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.set_ylabel("Score")
    ax5.set_title("Best vs Worst Response Comparison")
    ax5.legend()
    ax5.set_ylim(0, 1.1)

    for i, (b, w) in enumerate(zip(best_vals, worst_vals)):
        diff = b - w
        if abs(diff) > 0.1:
            ax5.annotate(
                f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}",
                (i, max(b, w) + 0.05),
                ha="center",
                fontsize=8,
                color="#2ecc71" if diff > 0 else "#e74c3c",
            )

    path = Path(output_dir) / "hypertune_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def create_hyperparameter_exploration(results, output_dir="."):
    if len(results) < 3:
        return None

    available_params = [
        p
        for p in [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "top_k",
        ]
        if p in results[0]["hyperparameters"]
    ]

    if len(available_params) < 2:
        return None

    df = pd.DataFrame(
        [
            {
                **r["hyperparameters"],
                "total_score": r["total_score"],
                "quality": r.get("quality_penalty", 1.0),
            }
            for r in results
        ]
    )

    n_params = len(available_params)
    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8))
    fig.suptitle("Hyperparameter Exploration", fontsize=14, fontweight="bold")

    if n_params == 1:
        axes = axes.reshape(2, 1)

    for i, param in enumerate(available_params):
        ax_scatter = axes[0, i]
        scatter = ax_scatter.scatter(
            df[param],
            df["total_score"],
            c=df["total_score"],
            cmap="RdYlGn",
            s=50,
            alpha=0.7,
            edgecolors="white",
        )

        corr = df[param].corr(df["total_score"])
        z = np.polyfit(df[param], df["total_score"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[param].min(), df[param].max(), 100)
        ax_scatter.plot(x_line, p(x_line), "--", color="#34495e", alpha=0.8)

        ax_scatter.set_xlabel(param)
        ax_scatter.set_ylabel("Score")
        ax_scatter.set_title(f"{param}\n(r = {corr:.2f})")
        ax_scatter.set_ylim(0, 1.1)

        ax_box = axes[1, i]
        bins = pd.qcut(df[param], q=min(4, len(df) // 2), duplicates="drop")
        df_temp = df.copy()
        df_temp["bin"] = bins
        sns.boxplot(
            data=df_temp,
            x="bin",
            y="total_score",
            ax=ax_box,
            hue="bin",
            palette="RdYlGn",
            legend=False,
        )
        ax_box.set_xlabel(f"{param} Range")
        ax_box.set_ylabel("Score")
        ax_box.tick_params(axis="x", rotation=45)
        ax_box.set_ylim(0, 1.1)

    plt.tight_layout()
    path = Path(output_dir) / "hyperparameter_exploration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def print_results(results, args):
    print(f"\n{'=' * 60}")
    print("HYPERTUNE RESULTS")
    print(f"{'=' * 60}")
    print(f"Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"Provider: {args.provider} | Model: {args.model or 'default'}")
    print(f"Iterations: {args.iterations} | Valid responses: {len(results)}")
    print(f"{'=' * 60}\n")

    top_n = min(args.top, len(results))

    for i, result in enumerate(results[:top_n], 1):
        quality = result.get("quality_penalty", 1.0)
        quality_warning = " ⚠️ LOW QUALITY" if quality < 0.5 else ""

        print(f"{'─' * 60}")
        print(f"RESPONSE #{i}{quality_warning}")
        print(f"{'─' * 60}")
        print(format_score_breakdown(result))
        print(f"\nHyperparameters: {format_hyperparameters(result['hyperparameters'])}")
        print(f"\nResponse:")

        if args.full:
            print(result["text"])
        else:
            print(truncate_text(result["text"]))
        print()

    if len(results) > top_n:
        print(
            f"[{len(results) - top_n} more responses not shown. Use --top N to see more.]\n"
        )

    print(f"{'=' * 60}")
    print("ANALYSIS")
    print(f"{'=' * 60}")

    if results:
        scores = [r["total_score"] for r in results]
        print(f"\nScore Statistics:")
        print(f"  Best:  {max(scores):.3f}")
        print(f"  Worst: {min(scores):.3f}")
        print(f"  Mean:  {np.mean(scores):.3f}")
        print(f"  Std:   {np.std(scores):.3f}")

        common_terms = extract_common_terms(results)
        if common_terms:
            print(f"\nTop Terms: {', '.join(term for term, _ in common_terms[:5])}")

        best = results[0]
        print(f"\nBest Hyperparameters:")
        for param, value in best["hyperparameters"].items():
            print(f"  {param}: {value}")


def list_providers():
    from hypertune.providers import ProviderFactory

    print("Available LLM Providers:")
    print("=" * 50)

    all_info = ProviderFactory.list_all_provider_info()
    for provider_name, info in all_info.items():
        if "error" in info:
            print(f"\n{provider_name.upper()}: {info['error']}")
            continue

        print(f"\n{provider_name.upper()}:")
        print(f"  Default model: {info['model']}")
        print("  Models:", ", ".join(info["available_models"][:5]))
        if len(info["available_models"]) > 5:
            print(f"    ... and {len(info['available_models']) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="HyperTune - LLM hyperparameter exploration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --prompt "Explain quantum computing" --iterations 5
  %(prog)s --prompt "Write a poem" --provider anthropic --output results.json
  %(prog)s --prompt "Summarize AI" --no-charts --top 5 --full
        """,
    )
    parser.add_argument("--prompt", help="Input prompt for generation")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations (default: 5)"
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "gemini", "openrouter"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model", help="Specific model (uses provider default if not specified)"
    )
    parser.add_argument("--output", "-o", help="Save full results to JSON file")
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of top results to display (default: 3)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full response text (default: truncated)",
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="Disable chart generation"
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and models",
    )

    args = parser.parse_args()

    if args.list_providers:
        list_providers()
        return

    if not args.prompt:
        parser.error("--prompt is required")

    print(f"Running HyperTune with {args.iterations} iterations...")

    try:
        ht = HyperTune(args.prompt, args.iterations, args.provider, args.model)
        results = ht.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No valid responses generated.", file=sys.stderr)
        sys.exit(1)

    print_results(results, args)

    if args.output:
        output_path = save_results_json(results, args, args.output)
        print(f"\nResults saved to: {output_path}")

    if not args.no_charts:
        dashboard = create_analysis_dashboard(results)
        exploration = create_hyperparameter_exploration(results)
        charts = [c for c in [dashboard, exploration] if c]
        if charts:
            print(f"\nCharts saved: {', '.join(str(c) for c in charts)}")


if __name__ == "__main__":
    main()
