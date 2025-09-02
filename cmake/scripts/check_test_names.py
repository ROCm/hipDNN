#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

class TestNameValidator:
    def __init__(self):
        self.valid_patterns = [
            # Standard: TestSuiteName.TestCaseName
            re.compile(r'^[A-Z][a-zA-Z0-9]*\.[A-Z][a-zA-Z0-9]*$'),
            # Parameterized: TestSuiteName/InstanceName.TestCaseName
            re.compile(r'^[A-Z][a-zA-Z0-9]*/[a-zA-Z0-9_]+\.[A-Z][a-zA-Z0-9]*$')
        ]
        
        # Controlled keywords
        self.keywords = {
            'test_types': ['Test', 'Integration'],
            'gpu': ['Gpu'],
            'datatypes': ['Bfp16', 'Fp16', 'Fp32', 'Fp64'],
            'shapes': ['Nhwc', 'Nchw', 'Ndhwc', 'Ncdhw']
        }
        
        self.valid_keywords = [kw for sublist in self.keywords.values() for kw in sublist]

        # General rules for the entire test name
        self.general_rules = [
            {'regex': re.compile(r'_'), 'message': "Test names should not contain underscores (gtest reserves _ for future use)"},
            {'regex': re.compile(r'\s'), 'message': "Test names should not contain spaces"},
            {'regex': re.compile(r'^\d'), 'message': "Test names should not start with a number"},
            {'regex': re.compile(r'^[^.]+$'), 'message': "Test name missing separator '.' between suite and test case"}
        ]

        # Rules specific to the test case
        self.case_rules = [
            {'regex': re.compile(r'^[a-z]'), 'message': "Test case name should start with an uppercase letter (PascalCase)"},
            {'validator': self._check_case_for_keywords, 'message': "Test case name should not contain keywords that belong in the suite name"}
        ]

        # Rules specific to the test suite
        self.suite_rules = [
            {'regex': re.compile(r'^[^A-Z]'), 'message': "Test suite name should start with an uppercase letter (PascalCase)"},
            {
                'validator': self._validate_suite_structure,
                'message': "Test suite name structure validation"
            }
        ]

    def _check_case_for_keywords(self, case_name: str) -> List[str]:
        """Validator to find disallowed keywords in a test case name."""
        found_keywords = [
            kw for kw in self.keywords['test_types'] + self.keywords['datatypes']
            if kw.lower() in case_name.lower()
        ]
        if found_keywords:
            return [f"Test case name should not contain keywords: {', '.join(found_keywords)}. These belong in the test suite name."]
        return []

    def _validate_suite_structure(self, suite_name: str) -> List[str]:
        """
        Validate the structure of a test suite name according to hipDNN conventions.
        Returns a list of issues found, or empty list if valid.
        """
        issues = []
        
        prefix_part = f"({'|'.join(self.keywords['test_types'])})?"
        gpu_part = f"({self.keywords['gpu'][0]})?"
        feature_part = r"[A-Z][a-zA-Z0-9]*"
        shapes_part = f"({'|'.join(self.keywords['shapes'])})?"
        datatypes_part = f"({'|'.join(self.keywords['datatypes'])})?"

        structure_regex = re.compile(
            f"^{prefix_part}{gpu_part}{feature_part}{shapes_part}{datatypes_part}$"
        )

        if not structure_regex.match(suite_name):
            issues.append("Suite name does not follow the structure: [Integration][Gpu]FeatureName[Shape][Datatype]")
        
        if 'Gpu' in suite_name:
            valid_position = suite_name.startswith('Gpu')
            for test_type in self.keywords['test_types']:
                if suite_name.startswith(test_type + 'Gpu'):
                    valid_position = True
                    break
            
            if not valid_position:
                issues.append("'Gpu' must be at the start of the suite name or immediately after a test type keyword")

        return issues

    def _parse_test_name(self, test_name: str) -> Optional[Tuple[str, str]]:
        """Parses a test name into its components."""
        if '.' not in test_name:
            return None
        
        suite_part, case_name = test_name.split('.', 1)
        suite_name = suite_part.split('/')[0]
        
        return suite_name, case_name


    def validate_test_name(self, test_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a single test name.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        for rule in self.general_rules:
            if rule['regex'].search(test_name) and ('validator' not in rule or rule['validator'](test_name)):
                issues.append(rule['message'])

        for keyword in self.valid_keywords:
            matches = re.findall(re.escape(keyword), test_name, re.IGNORECASE)
            
            # Check capitalization
            for match in matches:
                if match != keyword:
                    issues.append(f"Keyword '{match}' should be capitalized as '{keyword}'")
            
            # Check duplicates
            if len(matches) > 1:
                issues.append(f"Keyword '{keyword}' appears more than once.")

        parsed_name = self._parse_test_name(test_name)
        if parsed_name:
            suite_name, case_name = parsed_name

            # Validate suite using suite_rules
            for rule in self.suite_rules:
                if 'regex' in rule and rule['regex'].search(suite_name):
                    issues.append(rule['message'])
                if 'validator' in rule:
                    issues.extend(rule['validator'](suite_name))

            # Validate case
            for rule in self.case_rules:
                if 'regex' in rule and rule['regex'].search(case_name):
                    issues.append(rule['message'])
                if 'validator' in rule:
                    issues.extend(rule['validator'](case_name))

        is_valid = any(pattern.match(test_name) for pattern in self.valid_patterns)
        
        if not is_valid and not issues:
            issues.append("Test name doesn't match expected naming convention")
        
        return (is_valid and len(issues) == 0, issues)

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
    results: Dict[str, Tuple[bool, List[str]]] = {}
    
    for test_name in test_names:
        is_valid, issues = validator.validate_test_name(test_name)
        results[test_name] = (is_valid, issues)
        if not is_valid:
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
    
    for test_name, (is_valid, issues) in sorted(results.items()):
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
        print("  - No underscores anywhere (gtest reserves _ for future use)")
        print("  - No spaces, special characters, or leading numbers")
        print("\nKeyword Capitalization:")
        print("  - Integration, Gpu, Bfp16, Fp16, Fp32, Fp64, Nhwc, Nchw, Ndhwc, Ncdhw")
        print("  - All variants must use exact capitalization shown above")
        print("\nTest Suite Naming Structure:")
        print("  - Order: [Integration][Gpu]FeatureName[Shape][Datatype]")
        print("  - Integration tests: must start with 'Integration'")
        print("  - GPU tests: must include 'Gpu' (first, or after Integration)")
        print("  - Datatypes: Bfp16, Fp16, Fp32, Fp64 (at the end)")
        print("  - Shapes: Nhwc, Nchw, Ndhwc, Ncdhw (optional, only appear once in entire name)")
        print("\nExamples:")
        print("  - GpuBatchNorm.Forward")
        print("  - IntegrationGpuConvolutionNchwFp32.BackpropData")
        print("  - MemoryPool.Allocate")
        
    return 1 if args.strict else 0

if __name__ == '__main__':
    sys.exit(main())
