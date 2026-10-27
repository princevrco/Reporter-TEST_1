"""Build IPMVP_C_data.csv from alectra.csv and Open-Meteo daily mean temperatures.

Usage:
    py ingest.py --start 01-01-2025 --end 31-12-2025 --start-measure 01-07-2025
"""

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
SOURCE_CSV = HERE / "alectra.csv"
OUTPUT_CSV = HERE / "IPMVP_C_data.csv"
COLUMNS = ["date", "Summed Energy Used", "Heating Degree Days", "Cooling Degree Days", "Controlled"]
BASE_HEAT_TEMP_C = 22
BASE_COOL_TEMP_C = 15

# Default site: Promenade Shopping Centre, Vaughan, ON (Alectra service area).
DEFAULT_LAT = 43.807
DEFAULT_LON = -79.452
TIMEZONE = "America/Toronto"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def parse_dmy(text):
    try:
        return datetime.strptime(text, "%d-%m-%Y").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a valid dd-mm-yyyy date")


def parse_kwh(text):
    text = text.strip().replace(",", "")
    return float(text) if text else None


# Step 1: empty CSV with headers.
def create_empty_csv(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(COLUMNS)


# Step 2: read daily rows from alectra.csv and sum columns 2 and 3.
def read_daily_energy(path, start, end):
    """Return {date: summed kWh or None} for every date in [start, end] found in the file.

    The daily table starts after a header row whose first cell is "Date".
    The sum is left blank (None) when either meter has no reading for that day.
    """
    energy = {}
    in_daily_table = False
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0].strip() == "Date":
                in_daily_table = True
                continue
            if not in_daily_table:
                continue
            try:
                day = datetime.strptime(row[0].strip(), "%Y-%m-%d").date()
            except ValueError:
                continue
            if not (start <= day <= end):
                continue
            a, b = parse_kwh(row[1]), parse_kwh(row[2])
            energy[day] = None if a is None or b is None else round(a + b, 1)
    return energy


# Step 3: daily mean temperature from Open-Meteo, then HDD / CDD.
def fetch_mean_temps(url, lat, lon, start, end):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": TIMEZONE,
    }
    with urlopen(f"{url}?{urlencode(params)}", timeout=60) as resp:
        data = json.load(resp)
    daily = data["daily"]
    return {
        date.fromisoformat(d): t
        for d, t in zip(daily["time"], daily["temperature_2m_mean"])
        if t is not None
    }


def get_mean_temps(lat, lon, start, end):
    """Use the historical archive; fill any recent days it lacks from the forecast API."""
    temps = fetch_mean_temps(ARCHIVE_URL, lat, lon, start, end)
    missing = [d for d in daterange(start, end) if d not in temps]
    if missing:
        try:
            recent = fetch_mean_temps(FORECAST_URL, lat, lon, min(missing), max(missing))
            temps.update({d: t for d, t in recent.items() if d in missing})
        except Exception as exc:  # forecast API only covers recent months
            print(f"Warning: could not fill {len(missing)} day(s) from forecast API: {exc}", file=sys.stderr)
    return temps


def hdd(t):
    return max(BASE_HEAT_TEMP_C - t, 0)


def cdd(t):
    return max(t - BASE_COOL_TEMP_C, 0)


def daterange(start, end):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, type=parse_dmy, help="first date, dd-mm-yyyy")
    parser.add_argument("--end", required=True, type=parse_dmy, help="last date, dd-mm-yyyy")
    parser.add_argument("--start-measure", required=True, type=parse_dmy,
                        help="first measurement-period date, dd-mm-yyyy (Measured=1 on/after)")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="site latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="site longitude")
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be on or before --end")

    create_empty_csv(OUTPUT_CSV)

    energy = read_daily_energy(SOURCE_CSV, args.start, args.end)
    temps = get_mean_temps(args.lat, args.lon, args.start, args.end)

    rows_written = 0
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for day in daterange(args.start, args.end):
            if day not in energy:
                continue  # date not present in alectra.csv
            kwh = energy[day]
            t = temps.get(day)
            writer.writerow([
                day.isoformat(),
                "" if kwh is None else kwh,
                "" if t is None else round(hdd(t), 2),
                "" if t is None else round(cdd(t), 2),
                # Step 4: baseline (0) before --start-measure, measured (1) on/after.
                1 if day >= args.start_measure else 0,
            ])
            rows_written += 1

    blank_energy = sum(1 for v in energy.values() if v is None)
    blank_temp = sum(1 for d in energy if d not in temps)
    print(f"Wrote {rows_written} rows to {OUTPUT_CSV.name}")
    if blank_energy:
        print(f"  {blank_energy} row(s) have blank energy (a meter had no reading)")
    if blank_temp:
        print(f"  {blank_temp} row(s) have blank HDD/CDD (no temperature available)")


if __name__ == "__main__":
    main()
