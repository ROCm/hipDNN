#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

class TestNameValidator:
    def __init__(self):
        # Controlled keywords
        self.keywords = {
            'test_types': ['Test', 'Integration'],
            'gpu': ['Gpu'],
            'datatypes': ['Bfp16', 'Fp16', 'Fp32', 'Fp64'],
            'shapes': ['Nhwc', 'Nchw', 'Ndhwc', 'Ncdhw']
        }
        
        # Flattened list of all valid keywords for easy checks
        self.valid_keywords = [kw for sublist in self.keywords.values() for kw in sublist]

        # Regex to parse full test name into components
        self.full_name_re = re.compile(
            r'^(?P<suite>[A-Z][A-Za-z0-9]*)'
            r'(?:/(?P<instance>[A-Za-z0-9]+))?'
            r'\.(?P<case>(?:DISABLED_[A-Za-z0-9_]+|[A-Z][A-Za-z0-9]*))$'
        )

    def _validate_test_case(self, case_name: str) -> List[str]:
        """Fully validate a test case name according to hipDNN conventions."""
        issues = []
        
        dissallowed = self.keywords['test_types'] + self.keywords['datatypes'] + self.keywords['gpu']
        # Check for disallowed keywords
        found_keywords = [
            kw for kw in dissallowed
            if kw.lower() in case_name.lower()
        ]
        if found_keywords:
            issues.append(f"Test case name should not contain keywords: {', '.join(found_keywords)}. These belong in the test suite name.")
        
        return issues

    def _validate_suite_structure(self, suite_name: str) -> List[str]:
        """
        Validate the structure of a test suite name according to hipDNN test naming conventions.
        Returns a list of issues found, or empty list if valid.
        """
        issues = []
        
        prefix_part = f"({'|'.join(self.keywords['test_types'])})"
        gpu_part = f"({self.keywords['gpu'][0]})?"

        # feature_part = r"[A-Z][a-zA-Z0-9]*"

        disallowed_in_middle = self.keywords['test_types'] + self.keywords['gpu'] + self.keywords['datatypes']
        feature_part = f"(?!.*({'|'.join(disallowed_in_middle)}))[A-Z][a-zA-Z0-9]*"
        datatypes_part = f"({'|'.join(self.keywords['datatypes'])})?"

        structure_regex = re.compile(
            f"^{prefix_part}{gpu_part}{feature_part}{datatypes_part}$"
        )

        if not structure_regex.match(suite_name):
            issues.append("Suite name does not follow the structure: (Test|Integration)[Gpu?]FeatureName[Datatype?]")

        return issues

    def validate_test_name(self, test_name: str) -> List[str]:
        """
        Validate a single test name.
        Returns a list of issues found, or empty list if valid
        """
        issues: List[str] = []
        
        for keyword in self.valid_keywords:
            matches = re.findall(re.escape(keyword), test_name, re.IGNORECASE)
            
            # Check capitalization
            issues.extend([f"Keyword '{match}' should be capitalized as '{keyword}'" for match in matches if match != keyword])

            # Check duplicates
            if len(matches) > 1:
                issues.append(f"Keyword '{keyword}' appears more than once.") # Potentially useful to make test names more concise

        parsed_match = self.full_name_re.match(test_name)
        if not parsed_match:
            issues.append("Test name does not match expected PascalCase format 'TestSuite.TestCase' or 'TestSuite/Instance.TestCase' without special characters.")
            return issues

        suite_name = parsed_match.group('suite')
        case_name = parsed_match.group('case')

        issues.extend(self._validate_suite_structure(suite_name))
        issues.extend(self._validate_test_case(case_name))

        return issues

    def extract_test_names_from_ctest_json(self, json_path: Path) -> List[str]:
        """Extract test names from CTest JSON output."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            test_names = []
            if 'tests' in data:
                for test in data['tests']:
                    if 'name' in test:
                        test_names.append(test['name'])
            
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
        description='Validate test names against hipDNN rules'
    )

    parser.add_argument(
        '--ctest-json',
        type=Path,
        help='Path to CTest JSON output (from ctest --show-only=json-v1)',
        required=True
    )

    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit with non-zero status if any test names are invalid',
        default=False
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show all test names, not just invalid ones',
        default=False
    )

    parser.add_argument(
        '--guidelines',
        action='store_true',
        help='Show naming guidelines at end of output',
        default=False
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

    if args.guidelines:
        print("\nPlease update test names to follow hipDNN's naming conventions:")
        print("\nGeneral Rules:")
        print("  - Test suites: PascalCase (no underscores)")
        print("  - Test cases: PascalCase (no underscores)")
        print("  - Full format: TestSuite.TestCase")
        print("  - No underscores in test case names (except when starting with 'DISABLED_')")
        print("  - No underscores in test suite names")
        print("  - No spaces, special characters, or leading numbers")
        print("\nKeyword Capitalization:")
        print("  - Integration, Gpu, Bfp16, Fp16, Fp32, Fp64, Nhwc, Nchw, Ndhwc, Ncdhw")
        print("  - All variants must use exact capitalization shown above")
        print("\nTest Suite Naming Structure:")
        print("  - Order: (Test|Integration)[Gpu?]FeatureName[Datatype?]")
        print("  - TestType: REQUIRED - must be 'Test' or 'Integration' at the beginning")
        print("  - GPU tests: 'Gpu' must come immediately after the TestType")
        print("  - Datatypes: Bfp16, Fp16, Fp32, Fp64 (only at the end)")
        print("  - Shapes: Nhwc, Nchw, Ndhwc, Ncdhw (can appear anywhere in the name)")
        
    return 1 if args.strict else 0

if __name__ == '__main__':
    sys.exit(main())
