#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict


class TestNameValidator:

    KEYWORDS = {
        "test_types": ["Test", "Integration"],
        "gpu": ["Gpu"],
        "datatypes": ["Bfp16", "Fp16", "Fp32", "Fp64"],
        "shapes": ["Nhwc", "Nchw", "Ndhwc", "Ncdhw"],
    }

    FLATTENED_KEYWORDS = [kw for sublist in KEYWORDS.values() for kw in sublist]

    POSITIONAL_KEYWORDS = (
        KEYWORDS["test_types"] + KEYWORDS["datatypes"] + KEYWORDS["gpu"]
    )

    FULL_NAME_RE = re.compile(
        r"^(?:(?P<prefix>[A-Z][A-Za-z0-9]*)/)?"
        r"(?P<suite>[A-Z][A-Za-z0-9]*)"
        r"\.(?P<case>(?:DISABLED_[A-Za-z0-9_]+|[A-Z][A-Za-z0-9]*))"
        r"(?:/.*)?$"
    )

    def _validate_test_case(self, case_name: str) -> List[str]:
        """
        Validate a test case name.
        Returns a list of issues found, or empty list if valid.
        """
        issues = []

        # Check for disallowed positional keywords
        found_keywords = [kw for kw in self.POSITIONAL_KEYWORDS if kw in case_name]
        if found_keywords:
            issues.append(
                f"Test case name should not contain keywords: {', '.join(found_keywords)}. These belong in the test suite name."
            )

        return issues

    def _validate_suite_structure(self, suite_name: str) -> List[str]:
        """
        Validate the structure of a test suite name.
        Returns a list of issues found, or empty list if valid.
        """
        issues = []

        prefix_part = f"({'|'.join(self.KEYWORDS['test_types'])})"
        gpu_part = f"({self.KEYWORDS['gpu'][0]})?"
        feature_part = r"(?P<feature>[A-Z][a-zA-Z0-9]*?)"
        datatypes_part = f"({'|'.join(self.KEYWORDS['datatypes'])})?"

        structure_regex = re.compile(
            f"^{prefix_part}{gpu_part}{feature_part}{datatypes_part}$"
        )

        match = structure_regex.match(suite_name)

        if not match:
            issues.append(
                "Suite name does not follow the structure: (Test|Integration)[Gpu?]FeatureName[Datatype?]"
            )
            return issues

        feature_name = match.group("feature")

        for keyword in self.POSITIONAL_KEYWORDS:
            if keyword in feature_name:
                issues.append(
                    f"Keyword '{keyword}' is misplaced and should not be in the middle of the suite name."
                )

        return issues

    def validate_test_name(self, test_name: str) -> List[str]:
        """
        Validate a single full test name.
        Returns a list of issues found, or empty list if valid
        """
        issues = []

        parsed_match = self.FULL_NAME_RE.match(test_name)
        if not parsed_match:
            issues.append(
                "Test name does not match expected PascalCase format 'TestSuite.TestCase' or 'TestSuite/Instance.TestCase' without special characters."
            )
            return issues

        prefix = parsed_match.group("prefix") or ""
        suite_name = parsed_match.group("suite")
        case_name = parsed_match.group("case")

        for keyword in self.FLATTENED_KEYWORDS:
            matches = re.findall(
                re.escape(keyword), f"{prefix}/{suite_name}.{case_name}", re.IGNORECASE
            )

            valid_matches = [m for m in matches if m == keyword or m == keyword.upper()]
            # Check capitalization
            issues.extend(
                [
                    f"Keyword '{match}' should be capitalized as '{keyword}'"
                    for match in valid_matches
                    if match != keyword
                ]
            )

            # Check duplicates
            if len(valid_matches) > 1:
                issues.append(
                    f"Keyword '{keyword}' appears more than once."
                )  # Potentially useful to make test names more concise

        issues.extend(self._validate_suite_structure(suite_name))
        issues.extend(self._validate_test_case(case_name))

        return issues

    @staticmethod
    def extract_test_names_from_ctest_json(json_path: Path) -> List[str]:
        """Extract test names from CTest JSON output."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            test_names = []
            if "tests" in data:
                for test in data["tests"]:
                    if "name" in test:
                        test_names.append(test["name"].split("#")[0].strip())

            return test_names
        except FileNotFoundError:
            print(f"Error: CTest JSON file not found: {json_path}", file=sys.stderr)
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file {json_path}: {e}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"Unexpected error reading CTest JSON file: {e}", file=sys.stderr)
            return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate test names against hipDNN rules"
    )

    parser.add_argument(
        "--ctest-json",
        type=Path,
        help="Path to CTest JSON output (from ctest --show-only=json-v1)",
        required=True,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any test names are invalid",
        default=False,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all test names, not just invalid ones",
        default=False,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    validator = TestNameValidator()

    test_names = validator.extract_test_names_from_ctest_json(args.ctest_json)

    if not test_names:
        print("Warning: No test names found to validate", file=sys.stderr)
        return 0

    invalid_count = 0
    results: Dict[str, List[str]] = {}

    for test_name in test_names:
        issues = validator.validate_test_name(test_name)
        results[test_name] = issues
        if issues:
            invalid_count += 1

    print(f"\nTest Name Validation Report")
    print(f"{'=' * 60}")
    print(f"Total tests found: {len(test_names)}")
    print(f"Valid test names: {len(test_names) - invalid_count}")
    print(f"Invalid test names: {invalid_count}")

    if invalid_count == 0:
        return 0

    print(f"\n{'Test Name':<50} {'Status':<10}")
    print(f"{'-' * 60}")

    for test_name, issues in sorted(results.items()):
        is_valid = len(issues) == 0
        if not is_valid or args.verbose:
            status = "[PASS]" if is_valid else "[FAIL]"
            print(f"{test_name:<50} {status:<10}")
            if issues:
                for issue in issues:
                    print(f"  → {issue}")

    print(f"\nWarning: {invalid_count} test(s) have non-conforming names")
    print(
        " - For detailed hipDNN test naming rules, see: docs/CodingStyleAndNamingGuidelines.md\n"
    )

    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
