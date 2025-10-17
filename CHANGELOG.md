# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pre-commit configuration with black, ruff, mypy, and detect-secrets
- CODEOWNERS file for code review assignments
- Comprehensive repository audit and housekeeping
- Security scanning script (`scripts/scan_security.sh`)
- Comprehensive audit report (`audit_report.json`)
- PR materials and apply script for easy deployment
- Risk assessment and manual review documentation

### Changed
- Canonicalized configuration to `src/duoformer/config/`
- Canonicalized models to `src/duoformer/models/`
- Updated all import statements to use canonical paths
- Applied consistent code formatting with black and ruff
- Updated README.md to reflect canonical structure
- Fixed inconsistent path references in PROJECT_STRUCTURE.md

### Fixed
- Resolved critical mypy errors in core modules
- Fixed dataset loader type annotations
- Replaced legacy SwAV implementation with standard ResNet-50
- Added missing exports in training and exception modules
- Fixed documentation typos and inconsistencies

### Removed
- Stale `src/refactored_duoformer.egg-info/` directory
- Duplicate demo notebook and Dockerfile (archived)
- Top-level `config/` and `models/` directories (archived)
- `requirements.txt` from git tracking (now generated from `requirements.in`)

### Security
- Added secret scanning with detect-secrets
- Dependency audit and security hardening
- Documented CVE-2025-8869 (pip vulnerability)
- Created security audit report and recommendations

### DevOps
- Added pre-commit hooks for code quality
- Created CODEOWNERS file for review assignments
- Implemented security scanning in CI pipeline
- Added comprehensive CHANGELOG tracking

### Documentation
- Updated all documentation to reflect canonical structure
- Created comprehensive audit report
- Added PR materials and apply script
- Created risk assessment and manual review guide
