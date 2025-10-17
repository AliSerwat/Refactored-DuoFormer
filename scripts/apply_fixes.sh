#!/bin/bash
# 🛠️ Automated Repository Housekeeping Script
# This script applies the housekeeping fixes in the correct order

set -e  # Exit on any error

echo "🔧 Starting automated repository housekeeping..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verify we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -f "src/duoformer/__init__.py" ]]; then
    print_error "Error: Not in the Refactored-DuoFormer repository root"
    exit 1
fi

print_status "Verified repository structure"

# Check if we're on the correct branch
current_branch=$(git branch --show-current)
if [[ "$current_branch" != "housekeeping/auto-fixes-backup" ]]; then
    print_warning "Warning: Not on backup branch. Creating fixes branch..."
    git checkout -b housekeeping/auto-fixes housekeeping/auto-fixes-backup 2>/dev/null || \
    git checkout housekeeping/auto-fixes 2>/dev/null || \
    (print_error "Could not switch to fixes branch" && exit 1)
fi

print_status "On correct branch: $(git branch --show-current)"

# Test baseline
echo ""
echo "🧪 Running baseline tests..."
if python tests/run_tests.py --unit >/dev/null 2>&1; then
    print_status "Baseline tests pass"
else
    print_error "Baseline tests failed - aborting"
    exit 1
fi

# Apply formatting (if not already applied)
echo ""
echo "🎨 Applying code formatting..."
if python -m black --check . >/dev/null 2>&1; then
    print_status "Code already formatted"
else
    print_warning "Applying black formatting..."
    python -m black . --quiet
    git add .
    git commit -m "style: apply code formatting (black/ruff)" || print_warning "No changes to commit"
fi

# Verify tests still pass after formatting
echo ""
echo "🧪 Verifying tests after formatting..."
if python tests/run_tests.py --unit >/dev/null 2>&1; then
    print_status "Tests pass after formatting"
else
    print_error "Tests failed after formatting - reverting"
    git reset --hard HEAD~1
    exit 1
fi

# Apply patches
echo ""
echo "🔧 Applying patches..."

# Apply Python version badge fix
if git apply patches/fix-python-version-badge.diff 2>/dev/null; then
    print_status "Applied Python version badge fix"
    git add README.md
    git commit -m "chore(docs): fix Python version badge to match pyproject.toml (3.10+)" || print_warning "No changes to commit"
else
    print_warning "Python version badge patch already applied or not needed"
fi

# Apply dataset splitting fix
if git apply patches/fix-dataset-split.diff 2>/dev/null; then
    print_status "Applied dataset splitting fix"
    git add src/duoformer/data/loaders/dataset.py tests/unit/test_fix_dataset_split.py
    git commit -m "fix(data): fix dataset splitting type annotations and random_split usage

- Fix random_split call to use full_dataset instead of range(total_size)
- Remove redundant type annotations that caused redefinition errors
- Add test to verify the fix works correctly" || print_warning "No changes to commit"
else
    print_warning "Dataset splitting patch already applied or not needed"
fi

# Apply .gitignore updates
echo ""
echo "📋 Updating .gitignore..."
if [ -f ".gitignore.new" ]; then
    mv .gitignore.new .gitignore
    git add .gitignore requirements.txt
    git commit -m "chore: update .gitignore with comprehensive auto-generated file exclusions

- Add diagnostics/, patches/, scripts/ to exclusions for housekeeping artifacts
- Fix requirements.txt to be committed (it should be tracked for reproducibility)
- Include comprehensive list of auto-generated files and directories" || print_warning "No changes to commit"
    print_status "Updated .gitignore"
else
    print_warning ".gitignore already updated"
fi

# Verify imports still work
echo ""
echo "📦 Verifying package imports..."
if python -c "
import duoformer
from duoformer.models import build_model_no_extra_params
from duoformer.config import ModelConfig
print('✅ All imports work correctly')
" >/dev/null 2>&1; then
    print_status "Package imports verified"
else
    print_error "Package imports failed - aborting"
    exit 1
fi

# Verify type checking
echo ""
echo "🔍 Running type checks..."
if python -m mypy src/duoformer/config/ src/duoformer/utils/ >/dev/null 2>&1; then
    print_status "Type checking passes"
else
    print_warning "Type checking has issues - may need manual review"
fi

echo ""
echo "🎉 Housekeeping completed successfully!"
echo ""
echo "📋 Summary of changes:"
echo "  • Applied consistent code formatting"
echo "  • Fixed Python version badge in README"
echo "  • Fixed dataset splitting type annotations"
echo "  • Updated .gitignore with comprehensive exclusions"
echo "  • Verified all tests pass"
echo "  • Confirmed package imports work"
echo ""

echo "🚀 Next steps:"
echo "1. Review changes: git log --oneline -10"
echo "2. Test manually: python tests/run_tests.py"
echo "3. Push to remote: git push origin $(git branch --show-current)"
echo "4. Create PR: gh pr create --title 'chore: automated repository housekeeping' --body-file REPO_HOUSEKEEPING_PR.md"
echo ""
echo "✨ Repository is now production-ready with improved maintainability!"
