"""
Wine Review Data Validator
--------------------------
This script validates the raw wine review dataset before processing.
It ensures:
- Each record has 9 lines of "key: value"
- Every 10th line is blank (except possibly the last record)
- Keys appear in the same order across records
"""

def validate_wine_reviews(file_path: str) -> tuple[bool, str]:
    """
    Validates whether a wine review dataset has the expected format.

    Args:
        file_path (str): Path to the raw wine reviews text file.

    Returns:
        tuple[bool, str]: (Validation success flag, validation message)
    """
    expected_keys = [
        "wine/name",
        "wine/wineId",
        "wine/variant",
        "wine/year",
        "review/points",
        "review/time",
        "review/userId",
        "review/userName",
        "review/text"
    ]

    with open(file_path, "r", encoding="latin-1") as f:
        lines = [line.rstrip("\n") for line in f]

    # Handle case where last record has no trailing blank
    if (len(lines) + 1) % 10 == 0:
        lines.append("")

    if len(lines) % 10 != 0:
        return False, f"File length ({len(lines)}) is not aligned with 10-line blocks."

    for i in range(0, len(lines), 10):
        record_lines = lines[i:i + 9]
        blank_line = lines[i + 9]

        if i + 10 < len(lines) and blank_line != "":
            return False, f"Missing blank line after record {i // 10 + 1}."

        for j, (expected_key, line) in enumerate(zip(expected_keys, record_lines)):
            if not line.startswith(expected_key + ":"):
                return False, (
                    f"Record {i // 10 + 1}, line {j + 1} "
                    f"expected key '{expected_key}' but found '{line.split(':',1)[0]}'"
                )

    return True, "All records are valid and formatted correctly."


if __name__ == "__main__":
    valid, msg = validate_wine_reviews("cellartracker.txt")
    print(msg)
