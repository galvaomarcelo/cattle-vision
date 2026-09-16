"""
Cattle-Vision - Process Dataset

Processes the raw image metadata extracted by ExifTool and creates
the canonical clean dataset used by the remaining Cattle-Vision
pipeline.

Main functions:
    load_raw_metadata()       Load and validate the raw metadata CSV.
    process_row()             Create one clean dataset record.
    calculate_vertical_fov()  Calculate the vertical field of view.
    calculate_nadir_ground_coverage()
                              Estimate nadir ground coverage.
    calculate_nadir_gsd()     Calculate nadir-equivalent GSD.
    check_image_files()       Verify that referenced images exist.
    print_report()            Print a dataset processing summary.
    main()                    Run the complete dataset processing pipeline.

Outputs:
    data/metadata/dataset_clean.csv
    data/metadata/dataset_clean.json

The clean dataset is the canonical one-row-per-image dataset used by
subsequent analysis and visualization steps.
"""

from pathlib import Path
import math


from utils import (
    parse_float,
    parse_datetime,
    read_csv,
    write_csv,
    write_json,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"

INPUT_CSV = METADATA_DIR / "images_metadata.csv"

OUTPUT_CSV = METADATA_DIR / "dataset_clean.csv"
OUTPUT_JSON = METADATA_DIR / "dataset_clean.json"


# ============================================================
# COLUMNS TO KEEP
# ============================================================

KEEP_COLUMNS = [

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    "image_id",
    "FileName",
    "relative_path",

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    "DateTimeOriginal",

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    "ProductName",
    "UniqueCameraModel",

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    "ExifImageWidth",
    "ExifImageHeight",

    # --------------------------------------------------------
    # Exposure
    # --------------------------------------------------------

    "ExposureTime",
    "ISO",
    "ExposureCompensation",
    "WhiteBalanceCCT",

    # --------------------------------------------------------
    # Optics
    # --------------------------------------------------------

    "FNumber",
    "FocalLength",
    "FocalLengthIn35mmFormat",
    "FOV",

    # --------------------------------------------------------
    # GPS
    # --------------------------------------------------------

    "GPSLatitude",
    "GPSLongitude",
    "GPSAltitude",
    "RelativeAltitude",

    # --------------------------------------------------------
    # Gimbal attitude
    # --------------------------------------------------------

    "GimbalPitchDegree",
    "GimbalYawDegree",
    "GimbalRollDegree",

    # --------------------------------------------------------
    # Flight attitude / movement
    # --------------------------------------------------------

    "FlightPitchDegree",
    "FlightRollDegree",
    "FlightYawDegree",
    "FlightXSpeed",
    "FlightYSpeed",
    "FlightZSpeed",
]


# ============================================================
# DERIVED COLUMNS
# ============================================================

DERIVED_COLUMNS = [

    "camera_orientation",
    "off_nadir_angle_deg",
    "vertical_fov_deg",
    "nadir_ground_width_m",
    "nadir_ground_height_m",
    "nadir_equivalent_gsd_cm_px",
]


# ============================================================
# CAMERA ORIENTATION
# ============================================================

def calculate_off_nadir_angle(
    gimbal_pitch: float | str | None,
) -> float | None:
    """
    Calculate the camera off-nadir angle.

    DJI convention:

        -90° = nadir
          0° = horizontal

    Therefore:

        off_nadir = abs(pitch + 90)
    """

    pitch = parse_float(gimbal_pitch)

    if pitch is None:
        return None

    return abs(pitch + 90.0)


def classify_camera_orientation(
    gimbal_pitch: float | str | None,
) -> str | None:
    """
    Classify the image as NADIR or OBLIQUE.

    Images within 10 degrees of nadir are classified as NADIR.
    """

    off_nadir = calculate_off_nadir_angle(
        gimbal_pitch
    )

    if off_nadir is None:
        return None

    if off_nadir <= 10.0:
        return "NADIR"

    return "OBLIQUE"


# ============================================================
# FIELD OF VIEW
# ============================================================

def calculate_vertical_fov(
    horizontal_fov: float | str | None,
    image_width: float | str | None,
    image_height: float | str | None,
) -> float | None:
    """
    Calculate vertical FOV from horizontal FOV and image aspect ratio.

    Formula:

        vfov =
            2 * atan(
                tan(hfov / 2)
                * height / width
            )
    """

    hfov = parse_float(horizontal_fov)
    width = parse_float(image_width)
    height = parse_float(image_height)

    if (
        hfov is None
        or width is None
        or height is None
        or width <= 0
        or height <= 0
    ):
        return None

    hfov_rad = math.radians(hfov)

    vfov_rad = 2.0 * math.atan(
        math.tan(hfov_rad / 2.0)
        * height / width
    )

    return math.degrees(vfov_rad)


# ============================================================
# NADIR GROUND COVERAGE
# ============================================================

def calculate_nadir_ground_coverage(
    altitude: float | str | None,
    horizontal_fov: float | str | None,
    vertical_fov: float | str | None,
) -> tuple[float | None, float | None]:
    """
    Calculate approximate ground coverage assuming:

        - camera is NADIR
        - terrain is flat
        - altitude is relative altitude

    Returns:

        ground_width_m
        ground_height_m
    """

    altitude = parse_float(altitude)
    hfov = parse_float(horizontal_fov)
    vfov = parse_float(vertical_fov)

    if (
        altitude is None
        or hfov is None
        or vfov is None
        or altitude <= 0
    ):
        return None, None

    hfov_rad = math.radians(hfov)
    vfov_rad = math.radians(vfov)

    ground_width = (
        2.0
        * altitude
        * math.tan(hfov_rad / 2.0)
    )

    ground_height = (
        2.0
        * altitude
        * math.tan(vfov_rad / 2.0)
    )

    return ground_width, ground_height


# ============================================================
# NADIR-EQUIVALENT GSD
# ============================================================

def calculate_nadir_gsd(
    ground_width: float | None,
    image_width: float | str | None,
) -> float | None:
    """
    Calculate approximate nadir-equivalent GSD.

    Returns:

        meters / pixel
    """

    ground_width = parse_float(
        ground_width
    )

    image_width = parse_float(
        image_width
    )

    if (
        ground_width is None
        or image_width is None
        or image_width <= 0
    ):
        return None

    return ground_width / image_width


# ============================================================
# PROCESS ONE ROW
# ============================================================

def process_row(
    row: dict[str, str],
) -> dict:
    """
    Create one clean dataset record.

    Original selected metadata is copied first.
    Derived attributes are then calculated and added.
    """

    clean_row = {
        column: row.get(column, "")
        for column in KEEP_COLUMNS
    }

    # --------------------------------------------------------
    # Camera orientation
    # --------------------------------------------------------

    off_nadir = calculate_off_nadir_angle(
        row.get("GimbalPitchDegree")
    )

    orientation = classify_camera_orientation(
        row.get("GimbalPitchDegree")
    )

    clean_row["camera_orientation"] = (
        orientation or ""
    )

    clean_row["off_nadir_angle_deg"] = (
        round(off_nadir, 6)
        if off_nadir is not None
        else None
    )

    # --------------------------------------------------------
    # Vertical FOV
    # --------------------------------------------------------

    vertical_fov = calculate_vertical_fov(
        row.get("FOV"),
        row.get("ExifImageWidth"),
        row.get("ExifImageHeight"),
    )

    clean_row["vertical_fov_deg"] = (
        round(vertical_fov, 6)
        if vertical_fov is not None
        else None
    )

    # --------------------------------------------------------
    # Nadir ground coverage
    # --------------------------------------------------------

    ground_width, ground_height = (
        calculate_nadir_ground_coverage(
            row.get("RelativeAltitude"),
            row.get("FOV"),
            vertical_fov,
        )
    )

    clean_row["nadir_ground_width_m"] = (
        round(ground_width, 6)
        if ground_width is not None
        else None
    )

    clean_row["nadir_ground_height_m"] = (
        round(ground_height, 6)
        if ground_height is not None
        else None
    )

    # --------------------------------------------------------
    # Nadir-equivalent GSD
    # --------------------------------------------------------

    nadir_gsd = calculate_nadir_gsd(
        ground_width,
        row.get("ExifImageWidth"),
    )

    clean_row["nadir_equivalent_gsd_cm_px"] = (
        round(nadir_gsd * 100.0, 6)
        if nadir_gsd is not None
        else None
    )

    return clean_row


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_metadata() -> list[dict[str, str]]:
    """
    Load the raw metadata CSV.

    Raises:
        FileNotFoundError:
            If the input CSV does not exist.

        ValueError:
            If the CSV has no header or no rows.
    """

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{INPUT_CSV}"
        )

    rows = read_csv(INPUT_CSV)

    if not rows:
        raise ValueError(
            "Input CSV contains no rows."
        )

    return rows


# ============================================================
# VALIDATE RAW DATA
# ============================================================

def validate_columns(
    rows: list[dict[str, str]],
) -> None:
    """
    Verify that all required metadata columns exist.
    """

    if not rows:
        raise ValueError(
            "Cannot validate columns in an empty dataset."
        )

    fieldnames = rows[0].keys()

    missing = [
        column
        for column in KEEP_COLUMNS
        if column not in fieldnames
    ]

    if missing:

        print()
        print(
            "ERROR: Required metadata columns "
            "are missing:"
        )

        for column in missing:
            print(f"  - {column}")

        raise ValueError(
            "Raw metadata CSV does not contain "
            "all required columns."
        )


# ============================================================
# CHECK IMAGE FILES
# ============================================================

def check_image_files(
    rows: list[dict[str, str]],
) -> list[str]:
    """
    Check that all referenced image files exist.
    """

    missing = []

    for row in rows:

        filename = row.get(
            "FileName",
            "",
        )

        if not filename:
            missing.append(
                "(empty filename)"
            )
            continue

        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            missing.append(filename)

    return missing


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    raw_rows: list[dict[str, str]],
) -> list[dict]:
    """
    Process all raw metadata rows into clean dataset records.
    """

    return [
        process_row(row)
        for row in raw_rows
    ]


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    rows: list[dict],
) -> None:
    """
    Save the clean dataset as CSV and JSON.
    """

    fields = (
        KEEP_COLUMNS
        + DERIVED_COLUMNS
    )

    write_csv(
        OUTPUT_CSV,
        rows,
        fieldnames=fields,
    )

    write_json(
        OUTPUT_JSON,
        rows,
    )


# ============================================================
# REPORT HELPERS
# ============================================================

def get_numeric_values(
    rows: list[dict],
    column: str,
) -> list[float]:
    """
    Extract valid numeric values from a dataset column.
    """

    values = [
        parse_float(row.get(column))
        for row in rows
    ]

    return [
        value
        for value in values
        if value is not None
    ]


def print_report(
    raw_rows: list[dict],
    clean_rows: list[dict],
    missing_files: list[str],
) -> None:
    """
    Print a summary of the processed dataset.
    """

    print()
    print("=" * 70)
    print("CATTLE-VISION DATASET PROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print()
    print("FILES")
    print("-" * 70)

    print(f"Input:        {INPUT_CSV}")
    print(f"Clean CSV:    {OUTPUT_CSV}")
    print(f"Clean JSON:   {OUTPUT_JSON}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("DATASET")
    print("-" * 70)

    print(
        f"Input images:   {len(raw_rows)}"
    )

    print(
        f"Output images:  {len(clean_rows)}"
    )

    print(
        f"Original cols:  {len(KEEP_COLUMNS)}"
    )

    print(
        f"Derived cols:   {len(DERIVED_COLUMNS)}"
    )

    print(
        f"Total cols:     "
        f"{len(KEEP_COLUMNS) + len(DERIVED_COLUMNS)}"
    )

    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    orientation_counts = {}

    for row in clean_rows:

        orientation = row.get(
            "camera_orientation"
        )

        if orientation:
            orientation_counts[orientation] = (
                orientation_counts.get(
                    orientation,
                    0,
                )
                + 1
            )

    print()
    print("CAMERA ORIENTATION")
    print("-" * 70)

    print(
        f"NADIR:     "
        f"{orientation_counts.get('NADIR', 0)}"
    )

    print(
        f"OBLIQUE:   "
        f"{orientation_counts.get('OBLIQUE', 0)}"
    )

    # --------------------------------------------------------
    # Altitude
    # --------------------------------------------------------

    altitudes = get_numeric_values(
        clean_rows,
        "RelativeAltitude",
    )

    print()
    print("ALTITUDE")
    print("-" * 70)

    if altitudes:

        print(
            f"Range: "
            f"{min(altitudes):.2f} → "
            f"{max(altitudes):.2f} m"
        )

        print(
            f"Mean:  "
            f"{sum(altitudes) / len(altitudes):.2f} m"
        )

    # --------------------------------------------------------
    # GPS
    # --------------------------------------------------------

    latitudes = get_numeric_values(
        clean_rows,
        "GPSLatitude",
    )

    longitudes = get_numeric_values(
        clean_rows,
        "GPSLongitude",
    )

    print()
    print("GPS")
    print("-" * 70)

    if latitudes:

        print(
            f"Latitude:  "
            f"{min(latitudes):.7f} → "
            f"{max(latitudes):.7f}"
        )

    if longitudes:

        print(
            f"Longitude: "
            f"{min(longitudes):.7f} → "
            f"{max(longitudes):.7f}"
        )

    # --------------------------------------------------------
    # Optics
    # --------------------------------------------------------

    focal_lengths = get_numeric_values(
        clean_rows,
        "FocalLength",
    )

    fovs = get_numeric_values(
        clean_rows,
        "FOV",
    )

    print()
    print("OPTICS")
    print("-" * 70)

    if focal_lengths:

        print(
            f"Focal length: "
            f"{min(focal_lengths):.2f} → "
            f"{max(focal_lengths):.2f} mm"
        )

    if fovs:

        print(
            f"Horizontal FOV: "
            f"{min(fovs):.2f} → "
            f"{max(fovs):.2f}°"
        )

    # --------------------------------------------------------
    # GSD
    # --------------------------------------------------------

    gsd_values = get_numeric_values(
        clean_rows,
        "nadir_equivalent_gsd_cm_px",
    )

    print()
    print("NADIR-EQUIVALENT GSD")
    print("-" * 70)

    if gsd_values:

        print(
            f"Range: "
            f"{min(gsd_values):.2f} → "
            f"{max(gsd_values):.2f} cm/px"
        )

    # --------------------------------------------------------
    # Image files
    # --------------------------------------------------------

    print()
    print("IMAGE FILES")
    print("-" * 70)

    if not missing_files:

        print(
            f"All {len(clean_rows)} image files found."
        )

    else:

        print(
            f"Missing: {len(missing_files)}"
        )

        for filename in missing_files:
            print(f"  - {filename}")

    # --------------------------------------------------------
    # Derived columns
    # --------------------------------------------------------

    print()
    print("DERIVED COLUMNS")
    print("-" * 70)

    for column in DERIVED_COLUMNS:
        print(f"  - {column}")

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete dataset processing pipeline.
    """

    # --------------------------------------------------------
    # Load raw metadata
    # --------------------------------------------------------

    raw_rows = load_raw_metadata()

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    validate_columns(raw_rows)

    # --------------------------------------------------------
    # Process dataset
    # --------------------------------------------------------

    clean_rows = process_dataset(raw_rows)

    # --------------------------------------------------------
    # Check image files
    # --------------------------------------------------------

    missing_files = check_image_files(
        clean_rows
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_dataset(clean_rows)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_report(
        raw_rows,
        clean_rows,
        missing_files,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
