# plot_faithfulness.py
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="results/.../csv/<task>_node.csv or _neuron.csv")
    ap.add_argument("--title", default=None, help="Title for the plot")
    ap.add_argument("--out", default=None, help="Output filename (without extension)")
    ap.add_argument("--units", default=None, help="Total units text for x-label, e.g. '/32491'")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # expected columns: baseline, corrupted_baseline, requested_edges, edges, faithfulness
    baseline = df["baseline"].iloc[0]
    corrupted = df["corrupted_baseline"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(df["requested_edges"], [baseline]*len(df), linestyle="dotted", label="clean baseline")
    ax.plot(df["requested_edges"], [corrupted]*len(df), linestyle="dotted", label="corrupted baseline")
    ax.plot(df["edges"], df["faithfulness"], label="Faithfulness", color="tab:green")

    # labels & title
    units = f" ({args.units})" if args.units else ""
    ax.set_xlabel(f"Edges included{units}")
    ax.set_ylabel("logit_diff")
    title = args.title or os.path.splitext(os.path.basename(args.csv))[0].replace("_", " ")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    # outputs
    out_base = args.out or os.path.splitext(args.csv)[0]
    png = out_base + ".png"
    pdf = out_base + ".pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"Saved:\n  {png}\n  {pdf}")

if __name__ == "__main__":
    main()
