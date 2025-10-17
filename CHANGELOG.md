# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pre-commit configuration with black, ruff, mypy, and detect-secrets
- CODEOWNERS file for code review assignments
- Comprehensive repository audit and housekeeping

### Changed
- Canonicalized configuration to `src/duoformer/config/`
- Canonicalized models to `src/duoformer/models/`
- Updated all import statements to use canonical paths
- Applied consistent code formatting with black and ruff

### Fixed
- Resolved critical mypy errors in core modules
- Fixed dataset loader type annotations
- Replaced legacy SwAV implementation with standard ResNet-50
- Added missing exports in training and exception modules

### Removed
- Stale `src/refactored_duoformer.egg-info/` directory
- Duplicate demo notebook and Dockerfile (archived)
- Top-level `config/` and `models/` directories (archived)

### Security
- Added secret scanning with detect-secrets
- Dependency audit and security hardening

## [Previous Versions]

See git history for previous changes.
