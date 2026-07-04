"""
Generate a synthetic hospital encounter dataset.

SYNTHETIC DATA - NOT REAL PATIENTS. Every value in the output is randomly
generated from the distributions defined below. There is no real patient,
provider, or facility anywhere in this file or its output.

The dataset is shaped like a realistic hospital encounter / financial extract:
inpatient, emergency, observation, and outpatient-surgery encounters with
length of stay, charges and payments, MS-DRG and APR-DRG assignment, payer mix,
service departments, and a handful of length-of-stay outliers. It is meant for
building and demonstrating hospital analytics (length of stay, case mix, payer
mix, revenue cycle) without touching protected health information.

Usage:
    python generate.py                       # about 1,100 rows, CSV + XLSX
    python generate.py --scale 5             # roughly 5x as many rows
    python generate.py --seed 7 --format csv # different draw, CSV only
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference data (all fabricated)
# ---------------------------------------------------------------------------

PROVIDERS = [f"Dr. {n}" for n in [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson",
    "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee",
    "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez",
    "Lewis", "Robinson", "Walker",
]]

SPECIALTIES = [
    "Medicine", "Surgery", "Cardiology", "Orthopedics", "Neurology",
    "Pulmonology", "Gastroenterology", "Oncology", "Critical Care",
    "Emergency Medicine", "Hospitalist", "Infectious Disease",
]

PROVIDER_SPECIALTY = {p: SPECIALTIES[i % len(SPECIALTIES)] for i, p in enumerate(PROVIDERS)}

PROVIDER_GROUPS = [
    "Hospital Medicine Group", "Surgical Associates", "Cardiology Partners",
    "Orthopedic Institute", "Neuroscience Center", "Pulmonary & Critical Care",
    "GI Associates", "Oncology Network", "Emergency Physicians Group",
    "Infectious Disease Consultants",
]

PAYER_WEIGHTS = {
    "Medicare": 0.40, "Commercial": 0.30, "Medicaid": 0.15,
    "Self-Pay": 0.10, "Other": 0.05,
}

PLANS = {
    "Medicare": ["Medicare A", "Medicare B", "Medicare Advantage"],
    "Commercial": ["Commercial PPO", "Commercial HMO", "Commercial POS", "Commercial EPO"],
    "Medicaid": ["Medicaid FFS", "Medicaid Managed Care"],
    "Self-Pay": ["Self-Pay"],
    "Other": ["Workers Comp", "Tricare", "VA"],
}

# MS-DRG reference: (drg_code, drg_name, weight, specialty_hint, discharge_depts)
# DRG codes, names, and relative weights are public CMS reference values.
DRG_TABLE = [
    (190, "Chronic Obstructive Pulmonary Disease w MCC", 1.38, "Pulmonology", ["Pulmonary", "Med/Surg"]),
    (191, "Chronic Obstructive Pulmonary Disease w CC", 1.00, "Pulmonology", ["Pulmonary", "Med/Surg"]),
    (192, "Chronic Obstructive Pulmonary Disease w/o CC/MCC", 0.72, "Pulmonology", ["Pulmonary", "Med/Surg"]),
    (291, "Heart Failure & Shock w MCC", 1.74, "Cardiology", ["CCU", "Telemetry", "Med/Surg"]),
    (292, "Heart Failure & Shock w CC", 1.12, "Cardiology", ["Telemetry", "Med/Surg"]),
    (293, "Heart Failure & Shock w/o CC/MCC", 0.76, "Cardiology", ["Telemetry", "Med/Surg"]),
    (470, "Major Hip & Knee Joint Replacement", 1.74, "Orthopedics", ["Orthopedics"]),
    (689, "Kidney & Urinary Tract Infections w MCC", 1.29, "Medicine", ["Med/Surg"]),
    (690, "Kidney & Urinary Tract Infections w/o MCC", 0.82, "Medicine", ["Med/Surg"]),
    (871, "Septicemia or Severe Sepsis w/o MV >96 Hours w MCC", 2.28, "Critical Care", ["ICU", "Med/Surg"]),
    (872, "Septicemia or Severe Sepsis w/o MV >96 Hours w/o MCC", 1.29, "Medicine", ["Med/Surg"]),
    (377, "GI Hemorrhage w MCC", 1.70, "Gastroenterology", ["Med/Surg", "ICU"]),
    (378, "GI Hemorrhage w CC", 1.12, "Gastroenterology", ["Med/Surg"]),
    (480, "Hip & Femur Procedures Except Major Joint w MCC", 2.56, "Orthopedics", ["Orthopedics"]),
    (481, "Hip & Femur Procedures Except Major Joint w CC", 1.72, "Orthopedics", ["Orthopedics"]),
    (482, "Hip & Femur Procedures Except Major Joint w/o CC/MCC", 1.29, "Orthopedics", ["Orthopedics"]),
    (603, "Cellulitis w/o MCC", 0.88, "Medicine", ["Med/Surg"]),
    (65, "Intracranial Hemorrhage or Cerebral Infarction w MCC", 2.08, "Neurology", ["Neurology", "ICU"]),
    (66, "Intracranial Hemorrhage or Cerebral Infarction w CC", 1.22, "Neurology", ["Neurology"]),
    (194, "Simple Pneumonia & Pleurisy w CC", 1.01, "Pulmonology", ["Pulmonary", "Med/Surg"]),
    (193, "Simple Pneumonia & Pleurisy w MCC", 1.49, "Pulmonology", ["Pulmonary", "Med/Surg"]),
    (195, "Simple Pneumonia & Pleurisy w/o CC/MCC", 0.69, "Pulmonology", ["Pulmonary", "Med/Surg"]),
    (683, "Renal Failure w MCC", 1.64, "Medicine", ["Med/Surg", "ICU"]),
    (684, "Renal Failure w CC", 0.98, "Medicine", ["Med/Surg"]),
    (392, "Esophagitis, Gastroent & Misc Digest Disorders w/o MCC", 0.74, "Gastroenterology", ["Med/Surg"]),
    (309, "Cardiac Arrhythmia & Conduction Disorders w MCC", 1.35, "Cardiology", ["Telemetry", "CCU"]),
    (310, "Cardiac Arrhythmia & Conduction Disorders w CC", 0.84, "Cardiology", ["Telemetry"]),
    (247, "Perc Cardiovasc Proc w Drug-Eluting Stent w/o MCC", 2.04, "Cardiology", ["CCU", "Cardiac"]),
    (469, "Major Hip & Knee Joint Replacement w MCC", 3.17, "Orthopedics", ["Orthopedics"]),
    (917, "Poisoning & Toxic Effects of Drugs w MCC", 1.51, "Medicine", ["ICU", "Med/Surg"]),
]

# Primary diagnosis reference (public ICD-10-CM codes).
DX_MAP = {
    190: ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"),
    191: ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"),
    192: ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation"),
    291: ("I50.9", "Heart failure, unspecified"),
    292: ("I50.9", "Heart failure, unspecified"),
    293: ("I50.9", "Heart failure, unspecified"),
    470: ("M17.11", "Primary osteoarthritis, right knee"),
    689: ("N39.0", "Urinary tract infection, site not specified"),
    690: ("N39.0", "Urinary tract infection, site not specified"),
    871: ("A41.9", "Sepsis, unspecified organism"),
    872: ("A41.9", "Sepsis, unspecified organism"),
    377: ("K92.2", "Gastrointestinal hemorrhage, unspecified"),
    378: ("K92.2", "Gastrointestinal hemorrhage, unspecified"),
    480: ("S72.001A", "Fracture of unspecified part of neck of right femur"),
    481: ("S72.001A", "Fracture of unspecified part of neck of right femur"),
    482: ("S72.001A", "Fracture of unspecified part of neck of right femur"),
    603: ("L03.90", "Cellulitis, unspecified"),
    65:  ("I63.9", "Cerebral infarction, unspecified"),
    66:  ("I63.9", "Cerebral infarction, unspecified"),
    194: ("J18.9", "Pneumonia, unspecified organism"),
    193: ("J18.9", "Pneumonia, unspecified organism"),
    195: ("J18.9", "Pneumonia, unspecified organism"),
    683: ("N17.9", "Acute kidney failure, unspecified"),
    684: ("N17.9", "Acute kidney failure, unspecified"),
    392: ("K21.0", "Gastro-esophageal reflux disease with esophagitis"),
    309: ("I49.9", "Cardiac arrhythmia, unspecified"),
    310: ("I49.9", "Cardiac arrhythmia, unspecified"),
    247: ("I25.10", "Atherosclerotic heart disease of native coronary artery"),
    469: ("M17.11", "Primary osteoarthritis, right knee"),
    917: ("T50.901A", "Poisoning by unspecified drugs, accidental"),
}

# APR-DRG reference (public 3M APR-DRG code/description pairs).
APR_DRG_DATA = [
    (44, "Intracranial Hemorrhage"),
    (121, "Craniotomy"),
    (134, "Heart Failure"),
    (139, "Other Pneumonia"),
    (140, "Chronic Obstructive Pulmonary Disease"),
    (302, "Knee Joint Replacement"),
    (313, "Knee & Lower Leg Procedures"),
    (460, "Renal Failure"),
    (463, "Kidney & Urinary Tract Infections"),
    (720, "Septicemia & Disseminated Infections"),
    (249, "GI Hemorrhage"),
    (383, "Cellulitis"),
    (710, "Infectious & Parasitic Diseases"),
    (950, "Extensive Procedure Unrelated to Principal Diagnosis"),
]

# Encounter categories that a length-of-stay analysis typically excludes.
FILTER_CATEGORIES = {
    "Rehab": 30,
    "Newborn": 20,
    "Behavioral Health": 15,
    "Obstetrics": 40,
}

# Output column order.
COLUMN_ORDER = [
    "encounter_id", "visit_count", "patient_class", "account_class",
    "billing_status", "admit_datetime", "discharge_datetime",
    "discharge_department", "total_charges", "total_payments",
    "total_adjustments", "account_balance", "expected_collections",
    "admitting_provider", "attending_provider", "attending_specialty",
    "attending_provider_group", "primary_payer", "primary_plan",
    "ms_drg_code", "ms_drg_name", "apr_drg_code", "apr_drg_name",
    "apr_severity_of_illness", "billed_drg_code", "drg_case_type",
    "primary_dx_code", "primary_dx_description", "facility_campus",
    "ed_flag", "or_flag", "gi_flag", "ir_flag", "cancelled_flag",
    "observation_hours", "implant_charges", "ed_acuity_level",
    "inpatient_los_days", "readmit_flag", "ms_drg_weight",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_dates(start, end, n):
    """Return n random datetimes between start and end (minute resolution)."""
    start_ts = pd.Timestamp(start).timestamp()
    end_ts = pd.Timestamp(end).timestamp()
    ts = np.random.uniform(start_ts, end_ts, n)
    return pd.to_datetime(ts, unit="s").floor("min")


def generate_encounters(n, enc_type, patient_class, account_classes, los_mean, los_std):
    """Generate a block of encounters for a given patient type."""
    records = []
    adm_dates = random_dates("2024-01-01", "2025-12-31", n)

    for i in range(n):
        los_days = max(0.05, np.random.lognormal(np.log(los_mean), los_std))
        adm = adm_dates[i]
        disch = adm + pd.Timedelta(hours=los_days * 24)

        drg_code, drg_name, drg_weight, _spec_hint, dept_options = DRG_TABLE[np.random.randint(len(DRG_TABLE))]

        # Charges correlate with length of stay.
        base_charge = np.random.uniform(5000, 15000)
        total_charges = round(base_charge + los_days * np.random.uniform(3000, 8000), 2)
        total_charges = min(total_charges, 200000)
        total_charges = max(total_charges, 5000)

        total_payments = round(total_charges * np.random.uniform(0.60, 0.80), 2)
        total_adjustments = round(total_charges * np.random.uniform(0.05, 0.25), 2)
        account_balance = round(total_charges - total_payments - total_adjustments, 2)
        expected_collections = round(max(0, account_balance * np.random.uniform(0.5, 1.0)), 2)

        provider = np.random.choice(PROVIDERS)
        specialty = PROVIDER_SPECIALTY[provider]
        group = PROVIDER_GROUPS[SPECIALTIES.index(specialty) % len(PROVIDER_GROUPS)]

        payer = np.random.choice(list(PAYER_WEIGHTS.keys()), p=list(PAYER_WEIGHTS.values()))
        plan = np.random.choice(PLANS[payer])

        apr_code, apr_name = APR_DRG_DATA[np.random.randint(len(APR_DRG_DATA))]
        apr_soi = int(np.random.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.35, 0.15]))

        dx_code, dx_name = DX_MAP.get(drg_code, ("R69", "Illness, unspecified"))

        if enc_type == "Inpatient":
            dept = np.random.choice(dept_options)
        elif enc_type == "ED":
            dept = "ED"
        elif enc_type == "Observation":
            dept = np.random.choice(["Observation Unit", "Med/Surg", "Telemetry"])
        else:
            dept = np.random.choice(["PACU", "General Surgery", "GI Lab", "Interventional Radiology"])

        campus = np.random.choice(["Main Campus", "West Campus"], p=[0.80, 0.20])

        ed_flag = "Y" if enc_type == "ED" else ("Y" if np.random.random() < 0.15 else "N")
        or_flag = "Y" if enc_type == "Outpatient Surgery" else ("Y" if np.random.random() < 0.20 else "N")
        gi_flag = "Y" if (enc_type == "Outpatient Surgery" and "GI" in dept) else ("Y" if np.random.random() < 0.05 else "N")
        ir_flag = "Y" if (enc_type == "Outpatient Surgery" and "Interventional" in dept) else ("Y" if np.random.random() < 0.05 else "N")

        observation_hours = round(np.random.uniform(12, 72), 1) if enc_type == "Observation" else np.nan
        inpatient_los_days = round(los_days, 2) if enc_type == "Inpatient" else np.nan

        implant_charges = (
            round(np.random.uniform(5000, 40000), 2)
            if (drg_code in [470, 469, 480, 481, 482, 247] and np.random.random() < 0.7)
            else 0.0
        )

        ms_drg_weight = round(drg_weight * np.random.uniform(0.9, 1.1), 4)

        records.append({
            "encounter_id": None,  # assigned after shuffle
            "visit_count": int(np.random.randint(1, 8)),
            "patient_class": patient_class,
            "account_class": np.random.choice(account_classes),
            "billing_status": np.random.choice(
                ["Billed", "In Progress", "Closed", "Submitted"], p=[0.50, 0.20, 0.25, 0.05]
            ),
            "admit_datetime": adm,
            "discharge_datetime": disch,
            "discharge_department": dept,
            "total_charges": total_charges,
            "total_payments": total_payments,
            "total_adjustments": total_adjustments,
            "account_balance": account_balance,
            "expected_collections": expected_collections,
            "admitting_provider": provider,
            "attending_provider": provider,
            "attending_specialty": specialty,
            "attending_provider_group": group,
            "primary_payer": payer,
            "primary_plan": plan,
            "ms_drg_code": drg_code if enc_type == "Inpatient" else np.nan,
            "ms_drg_name": drg_name if enc_type == "Inpatient" else np.nan,
            "apr_drg_code": apr_code,
            "apr_drg_name": apr_name,
            "apr_severity_of_illness": apr_soi,
            "billed_drg_code": drg_code if enc_type == "Inpatient" else np.nan,
            "drg_case_type": "Medical" if drg_weight < 1.5 else "Surgical",
            "primary_dx_code": dx_code,
            "primary_dx_description": dx_name,
            "facility_campus": campus,
            "ed_flag": ed_flag,
            "or_flag": or_flag,
            "gi_flag": gi_flag,
            "ir_flag": ir_flag,
            "cancelled_flag": "Y" if np.random.random() < 0.03 else "N",
            "observation_hours": observation_hours,
            "implant_charges": implant_charges,
            "ed_acuity_level": int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.35, 0.30, 0.15])) if enc_type == "ED" else np.nan,
            "inpatient_los_days": inpatient_los_days,
            "readmit_flag": "Y" if np.random.random() < 0.08 else "N",
            "ms_drg_weight": ms_drg_weight if enc_type == "Inpatient" else np.nan,
        })

    return records


def add_outliers(records, n_high, n_extreme):
    """Inject length-of-stay outliers into inpatient records."""
    ip_idx = [i for i, r in enumerate(records) if r["patient_class"] == "Inpatient"]
    if not ip_idx:
        return

    n_high = min(n_high, len(ip_idx))
    chosen = np.random.choice(ip_idx, size=n_high, replace=False)
    for idx in chosen:
        new_los = np.random.uniform(15, 25)
        adm = records[idx]["admit_datetime"]
        records[idx]["discharge_datetime"] = adm + pd.Timedelta(days=new_los)
        records[idx]["inpatient_los_days"] = round(new_los, 2)
        records[idx]["total_charges"] = min(
            200000, round(records[idx]["total_charges"] + new_los * np.random.uniform(5000, 10000), 2)
        )
        records[idx]["total_payments"] = round(records[idx]["total_charges"] * np.random.uniform(0.60, 0.80), 2)

    remaining = [i for i in ip_idx if i not in set(chosen)]
    n_extreme = min(n_extreme, len(remaining))
    chosen2 = np.random.choice(remaining, size=n_extreme, replace=False) if remaining else []
    for idx in chosen2:
        new_los = np.random.uniform(25, 45)
        adm = records[idx]["admit_datetime"]
        records[idx]["discharge_datetime"] = adm + pd.Timedelta(days=new_los)
        records[idx]["inpatient_los_days"] = round(new_los, 2)
        records[idx]["total_charges"] = min(200000, round(new_los * np.random.uniform(6000, 10000), 2))
        records[idx]["total_payments"] = round(records[idx]["total_charges"] * np.random.uniform(0.55, 0.75), 2)


def generate_filter_cases(counts):
    """Generate encounter categories a length-of-stay analysis usually excludes."""
    filter_defs = {
        "Rehab": {
            "account_class": "Rehabilitation",
            "depts": ["Rehab Unit", "Physical Therapy"],
            "dx": ("Z96.641", "Presence of right artificial hip joint"),
            "los_mean": 12, "los_std": 0.4,
        },
        "Newborn": {
            "account_class": "Newborn",
            "depts": ["Nursery", "NICU"],
            "dx": ("Z38.00", "Single liveborn infant, delivered vaginally"),
            "los_mean": 3, "los_std": 0.5,
        },
        "Behavioral Health": {
            "account_class": "Behavioral Health",
            "depts": ["Psychiatry", "Behavioral Health Unit"],
            "dx": ("F32.9", "Major depressive disorder, single episode, unspecified"),
            "los_mean": 7, "los_std": 0.5,
        },
        "Obstetrics": {
            "account_class": "Obstetrics",
            "depts": ["Labor & Delivery", "Postpartum"],
            "dx": ("O80", "Encounter for full-term uncomplicated delivery"),
            "los_mean": 3, "los_std": 0.3,
        },
    }

    filter_records = []
    for cat, count in counts.items():
        defn = filter_defs[cat]
        adm_dates = random_dates("2024-01-01", "2025-12-31", count)
        for i in range(count):
            los_days = max(0.5, np.random.lognormal(np.log(defn["los_mean"]), defn["los_std"]))
            adm = adm_dates[i]
            disch = adm + pd.Timedelta(hours=los_days * 24)
            total_charges = min(200000, round(np.random.uniform(5000, 15000) + los_days * np.random.uniform(3000, 8000), 2))
            total_payments = round(total_charges * np.random.uniform(0.60, 0.80), 2)
            total_adjustments = round(total_charges * np.random.uniform(0.05, 0.25), 2)
            account_balance = round(total_charges - total_payments - total_adjustments, 2)
            provider = np.random.choice(PROVIDERS)
            specialty = PROVIDER_SPECIALTY[provider]
            group = PROVIDER_GROUPS[SPECIALTIES.index(specialty) % len(PROVIDER_GROUPS)]
            payer = np.random.choice(list(PAYER_WEIGHTS.keys()), p=list(PAYER_WEIGHTS.values()))
            plan = np.random.choice(PLANS[payer])
            apr_code, apr_name = APR_DRG_DATA[np.random.randint(len(APR_DRG_DATA))]

            filter_records.append({
                "encounter_id": None,
                "visit_count": int(np.random.randint(1, 5)),
                "patient_class": "Inpatient",
                "account_class": defn["account_class"],
                "billing_status": np.random.choice(["Billed", "Closed"], p=[0.6, 0.4]),
                "admit_datetime": adm,
                "discharge_datetime": disch,
                "discharge_department": np.random.choice(defn["depts"]),
                "total_charges": total_charges,
                "total_payments": total_payments,
                "total_adjustments": total_adjustments,
                "account_balance": account_balance,
                "expected_collections": round(max(0, account_balance * np.random.uniform(0.5, 1.0)), 2),
                "admitting_provider": provider,
                "attending_provider": provider,
                "attending_specialty": specialty,
                "attending_provider_group": group,
                "primary_payer": payer,
                "primary_plan": plan,
                "ms_drg_code": np.nan,
                "ms_drg_name": np.nan,
                "apr_drg_code": apr_code,
                "apr_drg_name": apr_name,
                "apr_severity_of_illness": int(np.random.choice([1, 2, 3, 4], p=[0.20, 0.40, 0.30, 0.10])),
                "billed_drg_code": np.nan,
                "drg_case_type": "Medical",
                "primary_dx_code": defn["dx"][0],
                "primary_dx_description": defn["dx"][1],
                "facility_campus": np.random.choice(["Main Campus", "West Campus"], p=[0.80, 0.20]),
                "ed_flag": "N",
                "or_flag": "Y" if cat == "Obstetrics" and np.random.random() < 0.3 else "N",
                "gi_flag": "N",
                "ir_flag": "N",
                "cancelled_flag": "N",
                "observation_hours": np.nan,
                "implant_charges": 0.0,
                "ed_acuity_level": np.nan,
                "inpatient_los_days": round(los_days, 2),
                "readmit_flag": "Y" if np.random.random() < 0.08 else "N",
                "ms_drg_weight": np.nan,
            })

    return filter_records


def build_dataset(scale=1.0):
    """Build the full synthetic encounter dataset as a DataFrame."""
    def s(n):
        return max(1, int(round(n * scale)))

    records = []
    records += generate_encounters(s(500), "Inpatient", "Inpatient", ["Inpatient"], los_mean=5.0, los_std=0.6)
    records += generate_encounters(s(200), "ED", "Emergency", ["Emergency", "Hospital Outpatient Services"], los_mean=0.3, los_std=0.5)
    records += generate_encounters(s(150), "Observation", "Observation", ["Observation", "Extended Recovery"], los_mean=1.5, los_std=0.4)
    records += generate_encounters(s(150), "Outpatient Surgery", "Outpatient Surgery", ["Hospital Outpatient Services", "Outpatient Surgery"], los_mean=0.5, los_std=0.4)

    add_outliers(records, n_high=s(20), n_extreme=s(10))
    records += generate_filter_cases({cat: s(n) for cat, n in FILTER_CATEGORIES.items()})

    np.random.shuffle(records)
    for i, rec in enumerate(records):
        rec["encounter_id"] = f"SYN-{100000 + i}"

    df = pd.DataFrame(records)[COLUMN_ORDER]

    # Use nullable integers so DRG/level codes render as "683", not "683.0",
    # and leave blank cells blank rather than showing NaN.
    for col in ["ms_drg_code", "billed_drg_code", "ed_acuity_level"]:
        df[col] = df[col].astype("Int64")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic hospital encounter dataset (no real patients).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default 42).")
    parser.add_argument("--scale", type=float, default=1.0, help="Multiply the base record counts by this factor (default 1.0, about 1,100 rows).")
    parser.add_argument("--out", default="synthetic_encounters", help="Output file stem, no extension (default 'synthetic_encounters').")
    parser.add_argument("--format", choices=["csv", "xlsx", "both"], default="both", help="Output format(s) to write.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    df = build_dataset(scale=args.scale)

    out = Path(args.out)
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)

    if args.format in ("csv", "both"):
        df.to_csv(out.with_suffix(".csv"), index=False)
    if args.format in ("xlsx", "both"):
        with pd.ExcelWriter(out.with_suffix(".xlsx"), engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="SYNTHETIC - NOT REAL", index=False)

    inpatient = df[df["patient_class"] == "Inpatient"]
    print(f"Generated {len(df)} synthetic encounters x {len(df.columns)} columns (seed {args.seed}, scale {args.scale}).")
    print(f"Patient class mix:\n{df['patient_class'].value_counts().to_string()}")
    print(f"Payer mix:\n{df['primary_payer'].value_counts().to_string()}")
    if not inpatient.empty:
        los = inpatient["inpatient_los_days"]
        print(f"Inpatient LOS: mean {los.mean():.2f}, median {los.median():.2f}, max {los.max():.2f} days")
        print(f"LOS outliers >15 days: {(los > 15).sum()}; >25 days: {(los > 25).sum()}")


if __name__ == "__main__":
    main()
