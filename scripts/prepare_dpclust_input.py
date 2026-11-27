from pathlib import Path
import gzip
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import traceback
import pandas as pd
import numpy as np

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Automatically detect the repository root
REPO_DIR = Path(__file__).resolve().parents[1]

# Input and output directories
DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = REPO_DIR / "results" / "preprocessing"

OUTPUT_DIR = RESULTS_DIR / "dpclust_inputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample to process
SAMPLE = "ICGC_PCA035_T01"

# Sample-specific preprocessing report (overwritten on each run)
REPORT_DIR = RESULTS_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / f"{SAMPLE}_preprocessing_report.txt"


def find_file(pattern):
    """Find exactly one file matching a pattern."""
    matches = [
        p for p in DATA_DIR.rglob(pattern)
        if p.is_file()
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one file for {pattern}, "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]


def normalize_chr(series):
    """Standardize chromosome identifiers."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"^chr", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )


def main():
    """Prepare DPClust input and write QC results."""
    # ============================================================
    # 2. LOCATE INPUT FILES
    # ============================================================

    vcf_candidates = [
        p for p in DATA_DIR.rglob(f"*{SAMPLE}*.vcf*")
        if p.is_file()
        and p.name.endswith((".vcf", ".vcf.gz"))
    ]

    if len(vcf_candidates) != 1:
        raise ValueError(f"Found VCF files: {vcf_candidates}")

    vcf_path = vcf_candidates[0]

    segments_path = find_file(
        f"{SAMPLE}_segments_clusters.txt"
    )

    ccf_path = find_file(
        f"{SAMPLE}_mutations_ccf.txt.gz"
    )

    cp_path = find_file(
        f"{SAMPLE}_confints_CP.txt"
    )

    print("Selected files:")
    print("VCF:", vcf_path)
    print("Segments:", segments_path)
    print("CCF:", ccf_path)
    print("Purity:", cp_path)


    # ============================================================
    # 3. LOAD INPUT FILES
    # ============================================================

    open_func = gzip.open if vcf_path.suffix == ".gz" else open

    with open_func(vcf_path, "rt", encoding="utf-8") as f:
        columns = next(
            line.strip().lstrip("#").split("\t")
            for line in f
            if line.startswith("#CHROM")
        )

    vcf = pd.read_csv(
        vcf_path,
        sep="\t",
        comment="#",
        header=None,
        names=columns,
        compression="infer"
    )

    segments = pd.read_csv(segments_path, sep="\t")

    ccf = pd.read_csv(
        ccf_path,
        sep="\t",
        compression="gzip"
    )

    cp = pd.read_csv(cp_path, sep="\t")

    # Sequenza purity estimates
    if len(cp) != 3:
        raise ValueError("Unexpected confints_CP format")

    purity = float(cp.iloc[1]["cellularity"])
    purity_lower = float(cp.iloc[0]["cellularity"])
    purity_upper = float(cp.iloc[2]["cellularity"])

    if not 0 < purity <= 1:
        raise ValueError(f"Invalid purity: {purity}")

    print("\nDataset summary:")
    print("VCF variants:", len(vcf))
    print("Sequenza CCF records:", len(ccf))
    print("Copy-number segments:", len(segments))

    print(
        f"Purity: {purity:.2f} "
        f"[{purity_lower:.2f}, {purity_upper:.2f}]"
    )


    # ============================================================
    # 4. EXTRACT SOMATIC MUTATIONS AND READ COUNTS
    # ============================================================

    # Select somatic SNVs with FILTER=PASS and valid read counts
    if "RECLASSIFICATION" in vcf.columns:
        vcf = vcf[
            vcf["RECLASSIFICATION"] == "somatic"
        ].copy()

    vcf = vcf[vcf["FILTER"] == "PASS"].copy()

    # DP4: REF forward, REF reverse, ALT forward, ALT reverse
    dp4 = vcf["INFO"].str.extract(
        r"(?:^|;)DP4=(\d+),(\d+),(\d+),(\d+)(?:;|$)"
    )

    dp4 = dp4.apply(pd.to_numeric, errors="coerce")

    vcf["WT.count"] = dp4[0] + dp4[1]
    vcf["mut.count"] = dp4[2] + dp4[3]

    vcf["chr"] = normalize_chr(vcf["CHROM"])

    vcf["end"] = pd.to_numeric(
        vcf["POS"], errors="coerce"
    )

    # Retain valid single nucleotide substitutions
    bases = ["A", "C", "G", "T"]

    vcf["REF"] = vcf["REF"].astype(str).str.upper()
    vcf["ALT"] = vcf["ALT"].astype(str).str.upper()

    snvs = vcf[
        vcf["REF"].isin(bases)
        & vcf["ALT"].isin(bases)
        & vcf["REF"].ne(vcf["ALT"])
    ].copy()

    snvs["mutation"] = snvs["REF"] + ">" + snvs["ALT"]

    snvs = snvs.dropna(
        subset=["end", "WT.count", "mut.count"]
    ).copy()

    snvs = snvs[
        (snvs["WT.count"] + snvs["mut.count"]) > 0
    ].copy()

    snvs["VAF"] = (
        snvs["mut.count"]
        / (snvs["WT.count"] + snvs["mut.count"])
    )

    print("\nSomatic SNVs:", len(snvs))


    # ============================================================
    # 5. MATCH VCF WITH SEQUENZA MUTATION CCF
    # ============================================================

    ccf["chr"] = normalize_chr(ccf["chromosome"])

    ccf["end"] = pd.to_numeric(
        ccf["position"], errors="coerce"
    )

    ccf["mutation"] = (
        ccf["mutation"].astype(str).str.strip().str.upper()
    )

    ccf_columns = [
        "chr",
        "end",
        "mutation",
        "CCF",
        "CCF.left",
        "CCF.right",
        "Mt",
        "CNt",
        "CNn",
        "F"
    ]

    ccf_subset = ccf[ccf_columns].copy()

    keys = ["chr", "end", "mutation"]

    if ccf_subset.duplicated(keys).any():
        raise ValueError("Duplicate mutation keys in Sequenza CCF")

    if snvs.duplicated(keys).any():
        raise ValueError("Duplicate mutation keys in VCF")

    merged = snvs.merge(
        ccf_subset,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True
    )

    match_rate = (
        (merged["_merge"] == "both").mean() * 100
    )

    print("\nMutation matching:")
    print(merged["_merge"].value_counts())
    print(f"Match rate: {match_rate:.2f}%")


    # ============================================================
    # 6. MATCH MUTATIONS WITH COPY-NUMBER SEGMENTS
    # ============================================================

    required_segment_columns = [
        "chromosome", "start.pos", "end.pos", "subCNt"
    ]

    missing_columns = [
        col for col in required_segment_columns
        if col not in segments.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing segment columns: {missing_columns}. "
            f"Available columns: {segments.columns.tolist()}"
        )

    segments["chr"] = normalize_chr(
        segments["chromosome"]
    )

    for col in ["start.pos", "end.pos", "subCNt"]:
        segments[col] = pd.to_numeric(
            segments[col], errors="coerce"
        )

    segments = segments.dropna(
        subset=["start.pos", "end.pos"]
    ).copy()

    if (segments["start.pos"] > segments["end.pos"]).any():
        raise ValueError("Invalid segment coordinates")

    # Initialize copy-number fields
    merged["subCNt"] = np.nan
    merged["segment_start"] = np.nan
    merged["segment_end"] = np.nan

    # Match each mutation to a segment on the same chromosome
    for chromosome, chrom_mutations in merged.groupby("chr"):

        chrom_segments = (
            segments[segments["chr"] == chromosome]
            .sort_values("start.pos")
            .reset_index(drop=True)
        )

        if chrom_segments.empty:
            continue

        starts = chrom_segments["start.pos"].to_numpy()
        ends = chrom_segments["end.pos"].to_numpy()

        # Check for overlapping segments
        if np.any(starts[1:] <= np.maximum.accumulate(ends)[:-1]):
            raise ValueError(
                f"Overlapping segments on chromosome {chromosome}"
            )

        positions = chrom_mutations["end"].to_numpy()

        # Find the segment starting immediately before each mutation
        segment_indices = np.searchsorted(
            starts, positions, side="right"
        ) - 1

        valid = segment_indices >= 0
        safe_indices = np.maximum(segment_indices, 0)

        # Check that the mutation is also before the segment end
        valid = valid & (positions <= ends[safe_indices])

        mutation_indices = chrom_mutations.index[valid]

        selected_segments = chrom_segments.iloc[
            segment_indices[valid]
        ]

        merged.loc[mutation_indices, "subCNt"] = (
            selected_segments["subCNt"].to_numpy()
        )

        merged.loc[mutation_indices, "segment_start"] = (
            selected_segments["start.pos"].to_numpy()
        )

        merged.loc[mutation_indices, "segment_end"] = (
            selected_segments["end.pos"].to_numpy()
        )

    print("\nCopy-number matching:")
    print("Mutations with subCNt:", merged["subCNt"].notna().sum())
    print("Mutations without subCNt:", merged["subCNt"].isna().sum())


    # ============================================================
    # 7. CALCULATE DPCLUST PARAMETERS
    # ============================================================

    numeric_columns = [
        "CCF", "Mt", "CNn", "CNt", "F", "subCNt"
    ]

    for col in numeric_columns:
        merged[col] = pd.to_numeric(
            merged[col], errors="coerce"
        )

    # Total tumour copy number at the mutation location
    merged["subclonal.CN"] = merged["subCNt"]

    # Mutation CCF
    merged["subclonal.fraction"] = merged["CCF"]

    # Mutation multiplicity
    merged["no.chrs.bearing.mut"] = merged["Mt"]

    # Mutation copy number: Sequenza-based estimate
    merged["mutation.copy.number"] = (
        merged["CCF"] * merged["Mt"]
    )

    # Alternative calculation using VCF read counts
    # Saved for quality control, not used as the main input
    merged["MCN_from_VAF"] = (
        (merged["VAF"] / purity)
        * (
            purity * merged["subclonal.CN"]
            + (1 - purity) * merged["CNn"]
        )
    )

    # Compare the two estimates
    merged["MCN_difference"] = (
        merged["mutation.copy.number"]
        - merged["MCN_from_VAF"]
    )

    # Compare the VCF and Sequenza allele frequencies
    merged["VAF_difference"] = (
        merged["VAF"] - merged["F"]
    )

    merged["phase"] = "unphased"


    # ============================================================
    # 8. VALIDATE DPCLUST INPUT
    # ============================================================

    dp_columns = [
        "chr",
        "end",
        "WT.count",
        "mut.count",
        "subclonal.CN",
        "mutation.copy.number",
        "subclonal.fraction",
        "no.chrs.bearing.mut",
        "phase"
    ]

    numeric_dp_columns = [
        col for col in dp_columns
        if col not in ["chr", "phase"]
    ]

    # Check for missing and non-finite values
    complete = merged[dp_columns].notna().all(axis=1)

    finite = np.isfinite(
        merged[numeric_dp_columns].to_numpy(dtype=float)
    ).all(axis=1)

    # Basic numerical checks
    valid_values = (
        (merged["end"] > 0)
        & (merged["WT.count"] >= 0)
        & (merged["mut.count"] >= 0)
        & (merged["subclonal.CN"] > 0)
        & (merged["mutation.copy.number"] >= 0)
        & (merged["subclonal.fraction"] >= 0)
        & (merged["no.chrs.bearing.mut"] > 0)
    )

    merged["valid_input"] = (
        complete & finite & valid_values
    )

    # Keep exclusion reasons for quality control
    merged["exclusion_reason"] = np.select(
        [
            merged["_merge"] == "left_only",
            merged["subCNt"].isna(),
            ~complete,
            ~finite,
            ~valid_values
        ],
        [
            "No matching Sequenza mutation",
            "Missing segment copy number",
            "Missing DPClust parameter",
            "Non-finite value",
            "Invalid numerical value"
        ],
        default=""
    )

    dp_input = merged.loc[
        merged["valid_input"], dp_columns
    ].copy()

    excluded = merged.loc[
        ~merged["valid_input"]
    ].copy()

    # Flag CCF > 1 for inspection, but do not truncate it
    merged["CCF_above_one"] = merged["CCF"] > 1

    if dp_input.empty:
        raise ValueError("No complete DPClust input rows")

    # DPClust identifies loci by chromosome and position
    if dp_input.duplicated(["chr", "end"]).any():
        raise ValueError(
            "Multiple mutations at the same genomic position"
        )

    # Convert positions and read counts to integers
    integer_columns = ["end", "WT.count", "mut.count"]

    for col in integer_columns:
        if (dp_input[col] % 1 != 0).any():
            raise ValueError(f"Non-integer values in {col}")

        dp_input[col] = dp_input[col].astype(int)


    # ============================================================
    # 9. EXPORT DPCLUST INPUT FILE
    # ============================================================

    input_filename = f"{SAMPLE}_dp_input.txt"

    dp_input.to_csv(
        OUTPUT_DIR / input_filename,
        sep="\t",
        index=False
    )


    # ============================================================
    # 10. SAVE QUALITY CONTROL FILES
    # ============================================================

    qc_columns = [
        "chr",
        "end",
        "REF",
        "ALT",
        "WT.count",
        "mut.count",
        "VAF",
        "F",
        "VAF_difference",
        "CNn",
        "CNt",
        "subCNt",
        "segment_start",
        "segment_end",
        "CCF",
        "CCF.left",
        "CCF.right",
        "Mt",
        "mutation.copy.number",
        "MCN_from_VAF",
        "MCN_difference",
        "CCF_above_one",
        "_merge",
        "valid_input",
        "exclusion_reason"
    ]

    merged[qc_columns].to_csv(
        OUTPUT_DIR / f"{SAMPLE}_matching_qc.csv",
        index=False
    )

    excluded.to_csv(
        OUTPUT_DIR / f"{SAMPLE}_excluded.csv",
        index=False
    )

    unmatched = merged[
        merged["_merge"] == "left_only"
    ]

    unmatched.to_csv(
        OUTPUT_DIR / f"{SAMPLE}_unmatched.csv",
        index=False
    )


    # ============================================================
    # 11. SUMMARY
    # ============================================================

    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)

    print("Sample:", SAMPLE)
    print("Purity:", purity)
    print("Somatic SNVs:", len(snvs))
    print("Matched mutations:", (merged["_merge"] == "both").sum())
    print("Unmatched mutations:", len(unmatched))
    print(f"Match rate: {match_rate:.2f}%")

    print("\nDPClust input:")
    print("Complete mutations:", len(dp_input))
    print("Excluded mutations:", len(excluded))

    print("\nExclusion reasons:")
    print(excluded["exclusion_reason"].value_counts())

    print("\nCCF values above 1:", merged["CCF_above_one"].sum())

    print("\nMutation copy number comparison:")
    print(
        merged.loc[
            merged["valid_input"], "MCN_difference"
        ].abs().describe()
    )

    print("\nPreview of DPClust input:")
    print(dp_input.head().to_string(index=False))

    print("\nOutput files saved in:")
    print(OUTPUT_DIR)

    print("\nGenerated input:", input_filename)
    print("Check master_file.txt before running DPClust.")

    # ============================================================
    # 12. INVESTIGATE MCN DISCREPANCIES
    # ============================================================

    valid = merged[merged["valid_input"]].copy()

    print("\nVAF comparison:")
    print(valid[["VAF", "F"]].describe())

    print("\nAbsolute VAF differences:")
    print(valid["VAF_difference"].abs().describe())

    # Calculate MCN using Sequenza VAF instead of VCF VAF
    valid["MCN_from_Sequenza_VAF"] = (
        (valid["F"] / purity)
        * (
            purity * valid["subclonal.CN"]
            + (1 - purity) * valid["CNn"]
        )
    )

    valid["MCN_Sequenza_difference"] = (
        valid["mutation.copy.number"]
        - valid["MCN_from_Sequenza_VAF"]
    )

    print("\nMCN difference using Sequenza VAF:")
    print(valid["MCN_Sequenza_difference"].abs().describe())

    print("\nDetailed mutation comparison:")
    print(
        valid[[
            "chr", "end",
            "VAF", "F",
            "CNn", "CNt", "subCNt",
            "CCF", "Mt",
            "mutation.copy.number",
            "MCN_from_VAF",
            "MCN_from_Sequenza_VAF"
        ]].head(10).to_string(index=False)
    )

    # ============================================================
    # 13. INVESTIGATE IMPLIED TUMOUR PURITY
    # ============================================================

    # Select diploid regions to simplify the calculation
    diploid = valid[
        (valid["CNn"] == 2)
        & (valid["CNt"] == 2)
        & (valid["subCNt"] == 2)
        & (valid["Mt"] == 1)
        & (valid["mutation.copy.number"] > 0)
    ].copy()

    # Infer purity from Sequenza mutation copy number
    diploid["implied_purity"] = (
        2 * diploid["VAF"]
        / diploid["mutation.copy.number"]
    )

    print("\nImplied purity in diploid regions:")
    print("Number of mutations:", len(diploid))
    print(diploid["implied_purity"].describe())

    print("\nReported purity:", purity)
    print("Median implied purity:",
          diploid["implied_purity"].median())

    # Inspect alternative Sequenza solutions
    solutions_path = find_file(
        f"{SAMPLE}_alternative_solutions.txt"
    )

    solutions = pd.read_csv(solutions_path, sep="\t")

    print("\nSequenza alternative solutions:")
    print(solutions.head(20).to_string(index=False))


if __name__ == "__main__":
    with REPORT_PATH.open("w", encoding="utf-8") as report:
        with redirect_stdout(report), redirect_stderr(report):
            print("=" * 60)
            print("DPCLUST PREPROCESSING REPORT")
            print("=" * 60)
            print("Sample:", SAMPLE)
            print("Execution date:", datetime.now().isoformat(timespec="seconds"))
            print("=" * 60)
            try:
                main()
            except Exception:
                print("\nSTATUS: FAILED")
                traceback.print_exc(file=report)
                raise SystemExit(1) from None
            else:
                print("\nSTATUS: PREPROCESSING COMPLETED")
                print("Input consistency must be reviewed before DPClust execution.")
