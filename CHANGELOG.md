# Changelog

All notable changes to mstables will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Nano Versioning](https://semver.org/):
- 1.0.1, 1.0.2, ..., 1.0.9 (patch increments)
- 1.0.9 → 1.1.0 (minor increment when patch reaches 9)
- 1.9.9 → 2.0.0 (major increment when minor reaches 9)

---

## [Unreleased]

### Added
- DandE.db database for tracking tests, errors, trial runs, and TODOs
- Nano versioning system (VersionManager)
- DDD test pyramid structure (Unit, Integration, E2E tests)
- Docker setup with PostgreSQL for financial data
- SQLite for DandE.db (development/operational tracking)
- Version tracking in CHANGELOG.md
- PDS v2.0.0 integration plan (PDS_V2_INTEGRATION_PLAN.md)
- DDD compliance merge plan (DDD_MERGE_PLAN.md)
- Comprehensive test framework (database, GPU, multi-machine, automation, Airflow)
- SLURM-compatible distributed computing system
- TODO scanner script for automated TODO tracking
- Configuration templates for machine settings (fleet_config.json.template, machine_config.json.template, data_paths.json.template)
- Machine abstraction system (removed hardcoded hostnames, IPs, usernames)
- Root directory organization (logs/, data/results/, scripts/ directories)

### Changed
- Updated project structure to support Docker architecture
- Abstracted all machine-specific information to config files
- Removed hardcoded hostnames, IP addresses, and usernames from codebase
- Organized root directory: moved logs, results, scripts to proper directories
- Updated .gitignore to exclude sensitive data and temporary files
- Moved internal documentation to docs/ directory (per PDS standards)

### Security
- Removed Project_Template, Simons Strategery, and data/ directories from git history
- Added sensitive directories to .gitignore
- Abstracted machine-specific information for privacy and security
- Configuration files now required (no hardcoded defaults)

### Fixed
- Database connection handling
- Test infrastructure setup
- Root directory file sprawl
- Git history cleanup (removed sensitive information from all commits)

### Planned (DDD Compliance)
- **Version 1.0.3**: Analyze and plan DDD merge (Project_Template, mstables_002)
- **Version 1.0.4**: Restructure for DDD (domain/application/infrastructure layers)
- **Version 1.0.5**: Merge Project_Template folder
- **Version 1.0.6**: Merge mstables_002 folder
- **Version 1.1.0**: Complete DDD compliance

---

## [1.0.2] - 2025-11-05

### Added
- Comprehensive test framework (database, GPU, multi-machine, automation, Airflow)
- SLURM-compatible distributed computing system
- TODO scanner script for automated TODO tracking
- Configuration templates for machine settings
- Machine abstraction system
- Root directory organization

### Changed
- Updated version to 1.0.2
- Abstracted machine-specific information from codebase
- Organized root directory structure

### Security
- Removed sensitive directories from git history
- Added configuration files to .gitignore
- Abstracted all hardcoded machine information

### Technical
- Git commits: cb7a57a, d075da3, d5d5b27
- Git branch: master

---


## [1.0.1] - 2025-01-XX

### Added
- Initial nano versioning setup
- DandE.db schema for test/error/trial/TODO tracking
- Docker Compose configuration
- PostgreSQL service for financial data
- SQLite for DandE.db (local development tracking)
- CHANGELOG.md with version tracking

### Changed
- Updated pyproject.toml version to 1.0.1

### Architecture
- **Financial Data**: PostgreSQL (Docker container)
  - Production-ready database
  - Better concurrent access
  - Scalable for large datasets
  - Port: 5432
- **DandE.db**: SQLite (local file in data/)
  - Development/operational metadata
  - Test results, errors, trials, TODOs
  - Simpler setup, no container needed
  - Tracked in version control

---

[Unreleased]: https://github.com/datatalking/mstables/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/datatalking/mstables/releases/tag/v1.0.1

