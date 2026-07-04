# synthetic-hospital-data

A single-file generator for a realistic but entirely synthetic hospital
encounter dataset. No real patients, providers, or facilities appear anywhere
in this project or its output. Every value is drawn from the random
distributions defined in `generate.py`.

The point is to have safe, shareable, PHI-free test data for hospital analytics:
length of stay, case mix, payer mix, and revenue cycle. It produces the shape of
a typical inpatient/outpatient encounter extract, so you can build and
demonstrate dashboards, models, or data pipelines without any protected health
information.

## What it produces

About 1,100 encounter rows by default (1,105 at scale 1.0, scalable), across four core encounter
types plus a set of categories that length-of-stay analyses usually exclude:

- Inpatient, Emergency, Observation, and Outpatient Surgery encounters
- Length of stay, with a realistic tail of high and extreme outliers
- Charges, payments, adjustments, account balance, and expected collections
- MS-DRG and APR-DRG assignment with severity of illness
- Payer mix (Medicare, Commercial, Medicaid, Self-Pay, Other) and plans
- Attending provider, specialty, and provider group
- Service departments, campus, and revenue-cycle activity flags
- Excluded categories: Rehab, Newborn, Behavioral Health, Obstetrics

DRG codes and weights, ICD-10-CM diagnosis codes, and APR-DRG descriptions are
public reference values. Everything else (patients, providers, dates, dollars)
is fabricated.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python generate.py
```

This writes `synthetic_encounters.csv` and `synthetic_encounters.xlsx` to the
current directory and prints a short summary (row count, patient-class mix,
payer mix, and length-of-stay statistics).

Options:

```bash
python generate.py --scale 5           # roughly 5x as many rows
python generate.py --seed 7            # a different but reproducible draw
python generate.py --format csv        # CSV only (or xlsx, or both)
python generate.py --out data/sample   # choose the output path stem
```

The same `--seed` always produces the same dataset.

## Schema

Every row is one encounter. Columns:

| Column | Description |
| --- | --- |
| `encounter_id` | Synthetic encounter identifier (`SYN-######`) |
| `visit_count` | Number of visits on the account |
| `patient_class` | Inpatient, Emergency, Observation, Outpatient Surgery |
| `account_class` | Finer account classification |
| `billing_status` | Billed, In Progress, Closed, Submitted |
| `admit_datetime` | Admission timestamp |
| `discharge_datetime` | Discharge timestamp |
| `discharge_department` | Discharging service department |
| `total_charges` | Gross charges |
| `total_payments` | Payments received |
| `total_adjustments` | Contractual and other adjustments |
| `account_balance` | Charges minus payments and adjustments |
| `expected_collections` | Expected remaining collections |
| `admitting_provider` | Admitting provider |
| `attending_provider` | Attending provider |
| `attending_specialty` | Attending provider specialty |
| `attending_provider_group` | Attending provider group |
| `primary_payer` | Primary payer |
| `primary_plan` | Primary plan |
| `ms_drg_code` | MS-DRG code (inpatient only) |
| `ms_drg_name` | MS-DRG description (inpatient only) |
| `apr_drg_code` | APR-DRG code |
| `apr_drg_name` | APR-DRG description |
| `apr_severity_of_illness` | APR severity of illness, 1 to 4 |
| `billed_drg_code` | Billed DRG (inpatient only) |
| `drg_case_type` | Medical or Surgical |
| `primary_dx_code` | Primary diagnosis, ICD-10-CM |
| `primary_dx_description` | Primary diagnosis description |
| `facility_campus` | Main Campus or West Campus |
| `ed_flag` | Emergency department activity (Y/N) |
| `or_flag` | Operating room activity (Y/N) |
| `gi_flag` | GI lab activity (Y/N) |
| `ir_flag` | Interventional radiology activity (Y/N) |
| `cancelled_flag` | Cancelled procedure (Y/N) |
| `observation_hours` | Observation hours (observation encounters) |
| `implant_charges` | Implant charges, where applicable |
| `ed_acuity_level` | ED acuity level 1 to 5 (ED encounters) |
| `inpatient_los_days` | Inpatient length of stay in days |
| `readmit_flag` | Readmission (Y/N) |
| `ms_drg_weight` | MS-DRG relative weight (inpatient only) |

A small example extract is in [`sample_output/`](sample_output/).

## Notes

This is a generator, not a statistical model of any specific hospital. The
distributions are hand-set to look plausible and to exercise common analytics
paths (a length-of-stay tail, a payer mix, excluded categories), not to match a
real facility's case mix. The column layout is a generic, industry-standard
encounter schema; it is not derived from any real health system's data export.

## License

MIT. See [LICENSE](LICENSE).
