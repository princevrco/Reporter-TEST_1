# Promenade energy analysis

Daily meter baseline regression, energy forecasting and measurement-and-verification
review tools. Version **0.3.3**. Python **3.11 or 3.12**. Runs on CPU.

The project analyzes all supplied meters independently. It compares measured energy
with a frozen historical baseline, forecasts baseline consumption for a selectable
period, and converts energy differences to CAD at a configurable flat rate.
It runs locally or in a Runpod Pod. It does not connect to CARMA or write BACnet commands.

## Analysis scope

| Output | Interpretation |
| --- | --- |
| Historical baseline | Estimated no-intervention energy from historical meter readings |
| Reporting comparison | Baseline minus measured kWh for matching eligible dates |
| CAD comparison | Signed kWh difference multiplied by the selected CAD/kWh rate |
| Three-month forecast | Future no-intervention baseline consumption; not optimized consumption |
| Optional zone analysis | Estimated thermal-mode durations and statistical meter associations, subject to timing, coverage and identification checks |

The two-file workflow uses a **calendar-only regression**. BMS data supplies timestamp
coverage context, not electrical consumption or verified runtime. Weather and other
operating changes are not adjusted in that workflow. A passed import or completed run
does not establish verified savings, a causal BMS effect, or standards certification.
The full CLI also supports supplied weather and independent covariates, as well as
daily or fixed 15/30/60-minute energy intervals. `run-all` supports daily data only.

The methodology references the supplied IPMVP December 1997 Option C document;
`config/protocol-reference.json` records that reference. Site commissioning, a reviewed
M&V plan, meter boundaries, adjustments and an uncertainty assessment remain required
before decision-grade savings claims. This package does not implement every protocol procedure.

## Install and verify

Extract or clone the repository, then change into the folder containing `pyproject.toml`.
Do not move an existing virtual environment between folders; recreate it instead.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,excel]'
python -m pytest -q
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

A synthetic demonstration is available without private data:

```bash
python examples/demo.py --output results/synthetic-demo --optimizer
```

Synthetic examples are not site results. The test extra includes pytest; the Excel
extra is needed only for Excel imports and workbook tests. Core CSV workflows need
only the base dependencies. Runtime dependencies are specified in `pyproject.toml`.

## Private input files

No real site data, trained models, generated reports, credentials or session captures
are included. Create `local/` inside the repository and place these files there:

```text
promenade-option-c/
  pyproject.toml
  config/promenade-test.json
  local/
    consumption_kwh.zip
    live_bms_history.csv
```

The CARMA ZIP must contain daily **Consumption kWh** CSV exports and their matching
quality JSON files, as produced by the separate exporter. Demand kW, cumulative
counters and monthly exports are not interchangeable with daily interval energy.
An extracted directory of those files is also accepted. The wide BMS CSV must have
a `timestamp` column with explicit UTC offsets. All meter IDs remain separate.

The data ZIP distributed privately in the earlier test can supply these two files.
Do not upload it, its contents, generated results or a virtual environment to GitHub.
`.gitignore` and `.dockerignore` exclude the standard private paths and data formats;
inspect staged files before committing. Custom folders and renamed files still need review.

## Promenade configuration

Use `config/promenade-test.json` for the supplied snapshots. Its assumptions are provisional:

- Full-control start: **October 2, 2026**, based on an approximate user report; exact start unconfirmed.
- Baseline end, exclusive: **July 24, 2026**, excluding uncertain early commissioning/control history.
- Daily timezone: `America/Toronto`; CARMA's daily convention still needs confirmation.
- Review flags remain false until the corresponding facts have been verified.

Edit `intervention_timestamp` and `baseline_end_exclusive` after confirming the actual
control history. The baseline end cannot be later than intervention. Earlier asset
optimization must also be excluded from untreated training. Do not change review flags
or loosen validation limits merely to obtain a positive savings result.

`config/site.json` is an example for the advanced CLI, including illustrative BMS
point mappings. It is not a commissioned all-asset configuration.

## Run the daily comparison and forecast

From the repository root with the environment active:

```bash
python scripts/test_promenade.py \
  --start 2026-10-02 --end 2026-10-16 \
  --forecast-start 2026-10-20 --forecast-end 2027-01-20 \
  --rate 0.12 --output results/promenade-test
```

Dates are site-local. **End dates are exclusive.** This example compares October 2–15
and forecasts October 20–January 19. Without forecast flags, the forecast starts on the
current site-local date and lasts three calendar months. Reporting defaults match the
original October snapshot; choose reporting dates appropriate to each new export.

The runner validates inputs, fits every meter separately, prints progress, and writes
the dashboard. Use a new output directory for each full run; existing directories are
not overwritten. Alternative input paths are accepted with `--meter` and `--bms`.
Missing readings are not filled with invented energy. All-zero, unidentifiable or
invalid meters may be blocked. Poor held-out performance is recorded as a review gate.

### Reuse existing models

To change forecast dates or the energy rate without repeating import/training:

```bash
python scripts/summarize_promenade.py \
  --results results/promenade-test \
  --forecast-start 2026-10-20 --forecast-end 2027-01-20 \
  --rate 0.12
```

This regenerates the overview, forecast files and summary CSV inside the selected
results directory. It reuses frozen model artifacts and does not retrain them with
new readings. Preserve a results copy if you need the previous forecast output.
Results from versions 0.3.1 and 0.3.2 can be used via an absolute `--results` path.

## Read the results

Download `results/promenade-test/index.html` and open it in a browser. It is a standalone
HTML file with embedded data and requires no web server. Select a meter to see:

- Reporting-period baseline and measured consumption.
- Future baseline forecast, distinguished from measured values.
- Signed complete-period kWh differences and CAD at the chosen rate.
- Validation findings and unverified virtual-total candidates.

| File | Contents |
| --- | --- |
| `overview.json` | Dashboard data, forecast window, rate and selected-boundary totals |
| `meter-summary.csv` | Per-meter forecast, kWh/CAD differences, validation and blocked reasons |
| `forecast-daily.csv` | Daily future baseline predictions by meter |
| `boundary-candidates.csv` | Existing virtual totals, kept separate and unverified |
| `site-total.json` | Total for an explicitly selected boundary, otherwise unavailable |
| `summary.json` | Original comparison run, meter folders and review gates |
| `meter-.../savings/` | Reporting-period JSON, CSV and HTML |
| `meter-.../future-forecast/` | Latest future-forecast JSON, CSV and HTML |
| `meter-import-audit.json`, `bms-audit.json` | Source provenance, import validation and BMS coverage |

Negative differences mean measured usage exceeded the predicted baseline. Complete-period
values remain unavailable when any required date is missing. Paired values refer only
to eligible matched dates; they are not complete-window savings. CAD uses the same
flat rate on both sides and excludes demand charges, taxes, fixed charges and other
bill components. Future optimized usage and future realized savings are unavailable.

### Whole-site totals

Predicting usage cannot resolve an unknown electrical boundary. The CARMA data includes
physical meters, submeters and virtual totals; summing all meters can double-count loads.
Until a non-overlapping boundary is identified, the site total remains unavailable.
The existing virtual totals are candidates, not automatically verified mall totals.

After verifying the appropriate meter or non-overlapping set, supply the actual IDs:

```bash
# Replace MAIN_METER_ID with the verified CARMA code.
python scripts/summarize_promenade.py --total-meters MAIN_METER_ID --rate 0.12
```

Multiple IDs are accepted after `--total-meters`. This is the operator's boundary
declaration, not automatic wiring verification. Missing or blocked selected meters
keep totals unavailable. Paired totals use common eligible dates across all selected
meters; complete totals require the full window for all of them. Model/site review
gates still apply to any numerical total.

## Optional optimizer and advanced CLI

```bash
option-c --help
option-c prepare-optimizer --help
option-c run-all --help
```

Canonical optimizer CSV records use `timestamp` (valid time), `issued_at`, `zone_id`
and `achieved_mode`. Confirmed mode meanings are -1 cooling, 0 thermal OFF and 1 heating.
Optional application/support flags, zone mappings and measured temperature/setpoint
bindings are configured under `optimizer` in the site config. Thermal OFF does not
imply zero fan or electrical load. Unknown application and deadband behavior remain uncertain.

The optional response regression learns from synchronized historical meter and eligible
optimizer records; no separate manual calibration file is required. It needs sufficient
coverage and independent ON/OFF variation (default 60 training days), held-out validation,
and reviewed equipment mappings. Associations are not causal zone savings. It does not
replace measured reporting consumption with an optimizer prediction.

The supplied legacy archive ending September 14 lacks the explicit timing contract
required by the importer and predates the October full-control period. It is not a
supported direct input. Its flattened trajectories must not be converted by guessing
zone order, forecast start, issue time or cadence.

## Runpod and optional container

For the first deployment, use a Pod with Python 3.11 or 3.12, clone/extract the code,
change into its repository root, install as above and upload private files under `local/`.
A GPU is unnecessary for this regression workflow. Download `index.html` to view results.

The Dockerfile starts the separate signed-job serverless handler, not the Pod test.
It installs the optional Runpod/Excel dependencies, runs as UID 10001, and requires:

- `OPTION_C_JOB_HMAC_KEY`: a private HMAC key of at least 32 characters.
- `OPTION_C_STORE_ROOT`: a persistent directory writable by UID 10001.

`examples/sign_job.py` signs jobs in a trusted backend. Never commit keys, signed site
jobs, session captures or credentials. The handler supports site-scoped train, forecast,
savings and post-model jobs with bounded inline payloads; large archives belong in
the Pod file workflow. Signed-job tests are offline. A live Docker/serverless deployment
has not been verified by this release; provision storage/permissions and test it before use.

## Verification and known limits

Release 0.3.3 passed 59 offline tests locally and built successfully as a Python wheel.
The configured GitHub Actions matrix has not yet run in your repository.

The previous real-data check discovered 247 meters: 209 produced comparison reports,
38 were blocked, and 159 fitted meters triggered the held-out error review gate.
The three-month check yielded complete forecasts for 208 meters. These are snapshot
results, not guaranteed counts for new inputs and not evidence of verified savings.
Calendar-only forecasting does not model seasonal heating/cooling demand; a three-month
winter forecast can be unreliable even when a numerical validation screen passes.

GitHub Actions runs the offline tests on Python 3.11 and 3.12 after push. No private
inputs, real CARMA access or BACnet writes are used by CI. Local checks and CI do not
certify an M&V plan or validate live Runpod infrastructure.
