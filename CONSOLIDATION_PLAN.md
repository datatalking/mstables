# Folder Consolidation Plan

## Analysis Summary

### Folders to Consolidate

| Folder | Size | Status | Action |
|--------|------|--------|--------|
| `notebooks/` | 756K | 1 notebook | **Merge** → `notebooks/analysis/` |
| `notebooks_TODO/` | 840K | 8 notebooks | **Extract & Archive** - Extract useful code, archive notebooks |
| `mstables_002/` | 48M | Duplicate/older version | **Extract & Archive** - Extract unique code, archive data |
| `CBOE_data/` | 8K | Data files | **Already integrated** - Data ingested, keep for reference |
| `Project_Template/` | 128K | PDS standards | **Extract & Archive** - Extract PDS docs, archive template |
| `non_working_code_py/` | 148K | Broken code | **Archive** - Move to archive |
| `Crypto_Statistical_Analysis/` | 3.4M | Crypto analysis | **Extract & Archive** - Extract useful code, archive notebooks |

---

## Integration Plan

### 1. **notebooks/** → Merge into unified notebooks structure

**Files:**
- `notebooks/data_overview.ipynb` - Active analysis notebook

**Action:**
- Keep as-is in `notebooks/`
- Rename to `notebooks/analysis/data_overview.ipynb`

---

### 2. **notebooks_TODO/** → Extract useful code, archive notebooks

**Files:**
- `data_overview_002.ipynb` - Data analysis (duplicate functionality)
- `data_overview_003.ipynb` - Data analysis (duplicate functionality)
- `data_overview_new.ipynb` - Data analysis (duplicate functionality)
- `data_overview.ipynb` - Data analysis (duplicate)
- `predicting-s-p500-index-linearreg-randomforests.ipynb` - ML model (extract to `src/models/`)
- `quantstats_test.ipynb` - QuantStats testing (extract to tests)
- `morningstar_test.ipynb` - Morningstar testing (extract to tests)
- `FMP_backup_analyzer_2.ipynb` - Backup analyzer (archive)

**Action:**
1. Extract `data_overview_*.ipynb` code → `src/model/data_overview_v2.py` (if useful)
2. Extract ML model code → `src/models/regression/s_p500_predictor.py`
3. Extract test code → `tests/`
4. Archive all notebooks → `archive/notebooks_TODO/`

---

### 3. **mstables_002/** → Extract unique code, archive data

**Unique Code to Extract:**
- `src/liquidity_assumptions.py` → `src/domain/liquidity_assumptions.py`
- `src/regime_changes.py` → `src/domain/regime_changes.py`
- `src/market_assertions.py` → `src/domain/market_assertions.py`
- `src/dataframes.py` → Compare with existing, merge differences
- `config.py` → Review for useful config patterns

**Data to Archive:**
- `data/mstables_001.sqlite` → `archive/mstables_002/data/`
- `data/mstables.sqlite` (if different from main)
- `data/sample_rules_output.csv` → Review, keep if useful

**Input Files:**
- Compare `input/` with main `input/` directory
- Merge unique files, archive duplicates

**Action:**
1. Extract unique Python modules to appropriate DDD layers
2. Compare and merge `dataframes.py` with existing code
3. Archive data files → `archive/mstables_002/`
4. Archive notebooks → `archive/mstables_002/notebooks/`

---

### 4. **CBOE_data/** → Already integrated, archive reference data

**Status:**
- Data already ingested via `src/cboe/fetcher.py`
- Files are reference/historical data

**Action:**
- Move to `archive/CBOE_data/` for reference
- Keep structure for historical reference

---

### 5. **Project_Template/** → Extract PDS docs, archive template

**To Extract:**
- `CRITICAL_LESSONS_LEARNED.md` → `docs/PDS/CRITICAL_LESSONS_LEARNED.md`
- `UNIVERSAL_CODEBASE_IMPROVEMENT_FRAMEWORK.md` → `docs/PDS/`
- `scripts/todo_tracker.py` → Already have `scripts/todo_scanner.py` (compare and merge)
- `pds_compliance_checker.py` → `src/infrastructure/pds_compliance_checker.py`
- `universal_codebase_improver.py` → `src/infrastructure/universal_codebase_improver.py`
- `master_implementation_script.py` → `scripts/master_implementation.py`

**To Archive:**
- Template structure → `archive/Project_Template/`
- Test examples → `archive/Project_Template/tests/`

**Action:**
1. Extract documentation to `docs/PDS/`
2. Extract useful Python modules to `src/infrastructure/`
3. Archive template structure → `archive/Project_Template/`

---

### 6. **non_working_code_py/** → Archive

**Status:**
- Broken code that doesn't work
- Notebooks with errors

**Action:**
- Move entire folder → `archive/non_working_code_py/`
- Keep for reference but don't integrate

---

### 7. **Crypto_Statistical_Analysis/** → Extract useful code, archive notebooks

**Files:**
- Multiple crypto analysis notebooks
- `ANDREW_SECRETS.md` - Contains sensitive info (should be in .gitignore)

**To Extract:**
- Useful crypto analysis functions → `src/analysis/crypto/`
- API comparison docs → `docs/apis/`

**To Archive:**
- All notebooks → `archive/Crypto_Statistical_Analysis/`
- Sensitive files should be in .gitignore

**Action:**
1. Extract useful analysis functions
2. Archive notebooks → `archive/Crypto_Statistical_Analysis/`
3. Add `ANDREW_SECRETS.md` to .gitignore if not already

---

## Implementation Steps

### Phase 1: Extract Unique Code (High Priority)
1. Extract `mstables_002/src/liquidity_assumptions.py` → `src/domain/liquidity_assumptions.py`
2. Extract `mstables_002/src/regime_changes.py` → `src/domain/regime_changes.py`
3. Extract `mstables_002/src/market_assertions.py` → `src/domain/market_assertions.py`
4. Extract `Project_Template/pds_compliance_checker.py` → `src/infrastructure/pds_compliance_checker.py`
5. Extract `Project_Template/universal_codebase_improver.py` → `src/infrastructure/universal_codebase_improver.py`

### Phase 2: Merge Duplicate Code
1. Compare `mstables_002/src/dataframes.py` with existing dataframes code
2. Merge differences
3. Compare `Project_Template/scripts/todo_tracker.py` with `scripts/todo_scanner.py`
4. Merge useful features

### Phase 3: Archive Everything Else
1. Create `archive/` directory structure
2. Move folders to archive
3. Update .gitignore
4. Remove from git tracking

### Phase 4: Clean Up
1. Update imports
2. Fix broken references
3. Update tests
4. Update documentation

---

## Archive Structure

```
archive/
├── notebooks_TODO/          # Archived notebooks
├── mstables_002/            # Archived older version
│   ├── data/                # Archived databases
│   ├── notebooks/           # Archived notebooks
│   └── input/               # Archived input files
├── CBOE_data/               # Archived reference data
├── Project_Template/        # Archived template structure
├── non_working_code_py/    # Archived broken code
└── Crypto_Statistical_Analysis/  # Archived crypto notebooks
```

---

## Benefits

1. **Cleaner Structure** - Only active code in main directories
2. **Better Organization** - Code in appropriate DDD layers
3. **Reduced Confusion** - No duplicate/old versions
4. **Preserved History** - Archived for reference
5. **Easier Maintenance** - Clear separation of active vs archived

---

## Size Reduction

- **Before**: ~53MB of scattered folders
- **After**: ~10MB archived, ~500KB extracted useful code
- **Savings**: Cleaner structure, better organization

---

## Next Steps

1. Create archive directory structure
2. Extract unique code modules
3. Archive folders
4. Update .gitignore
5. Test that everything still works
6. Commit changes

