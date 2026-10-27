"""Chart measured daily energy against the regression estimate for the last three months.

Reads regression_fitted.csv (written by regression.py) and saves
chart_last_3_months.png.

Usage:
    py chart_last_3_months.py [--months 3]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from ingest import BASE_COOL_TEMP_C, BASE_HEAT_TEMP_C

HERE = Path(__file__).resolve().parent
FITTED_CSV = HERE / "regression_fitted.csv"
OUT_PNG = HERE / "chart_last_3_months.png"

# Default dataviz palette (light mode).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
MEASURED = "#2a78d6"   # categorical slot 1
ESTIMATE = "#eb6834"   # categorical slot 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--months", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(FITTED_CSV, parse_dates=["date"])
    end = df["date"].max()
    df = df[df["date"] > end - pd.DateOffset(months=args.months)].reset_index(drop=True)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(df["date"], df["E"], color=MEASURED, lw=2, label="Measured", zorder=3)
    ax.plot(df["date"], df["fitted"], color=ESTIMATE, lw=2, ls=(0, (4, 2)),
            label="Regression estimate", zorder=3)

    # Start of the controlled period.
    ctrl = df.loc[df["Controlled"] == 1, "date"]
    if not ctrl.empty and ctrl.min() > df["date"].min():
        start = ctrl.min()
        ax.axvline(start, color=MUTED, lw=1, ls=":", zorder=2)
        ax.annotate(f"Controlled period starts {start:%d %b}", xy=(start, 1), xycoords=("data", "axes fraction"),
                    xytext=(6, -4), textcoords="offset points", va="top", color=TEXT_2, fontsize=9)

    # Direct labels at the right end of each line.
    last = df.iloc[-1]
    for value, text, other in ((last["E"], "Measured", last["fitted"]),
                               (last["fitted"], "Estimate", last["E"])):
        offset = 8 if value >= other else -8
        ax.annotate(text, xy=(last["date"], value), xytext=(6, offset), textcoords="offset points",
                    va="center", color=TEXT, fontsize=9)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("kWh per day", color=TEXT_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, length=0, pad=6)
    ax.set_xlim(df["date"].min(), df["date"].max() + pd.Timedelta(days=6))

    mean_gap = (df["E"] - df["fitted"]).mean()
    fig.suptitle("Measured vs regression estimate, daily electricity", x=0.06, ha="left",
                 fontsize=14, color=TEXT, fontweight="bold")
    ax.set_title(f"{df['date'].min():%d %b %Y} to {end:%d %b %Y}  ·  degree-day bases: heating "
                 f"{BASE_HEAT_TEMP_C} °C, cooling {BASE_COOL_TEMP_C} °C  ·  mean measured − estimate: "
                 f"{mean_gap:+,.0f} kWh/day", loc="left", fontsize=9.5, color=TEXT_2, pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
              labelcolor=TEXT, handlelength=2.5, borderaxespad=0.2)

    fig.tight_layout()
    fig.savefig(OUT_PNG, facecolor=SURFACE)
    print(f"Saved {OUT_PNG.name} ({len(df)} days, mean measured - estimate {mean_gap:+,.0f} kWh/day)")


if __name__ == "__main__":
    main()
