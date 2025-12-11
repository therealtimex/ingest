# Release Process

This repository uses automated CI/CD workflows for releases.

## How It Works

1. **Update Version** - Change the version in `setup.py`
2. **Commit & Push** - Push to main branch
3. **Auto-Release** - GitHub Actions automatically:
   - Detects version change
   - Creates a git tag (e.g., `v1.0.0`)
   - Creates a GitHub release with changelog
   - Publishes to PyPI

## Step-by-Step Release

### 1. Update Version

Edit `setup.py`:

```python
setup(
    name="ingest-cli",
    version="1.0.1",  # Update this
    ...
)
```

### 2. Commit and Push

```bash
git add setup.py
git commit -m "Bump version to 1.0.1"
git push origin main
```

### 3. Automatic Process

The workflows will automatically:
- ✅ Create tag `v1.0.1`
- ✅ Create GitHub release
- ✅ Build package
- ✅ Publish to PyPI

You can monitor progress at:
- https://github.com/therealtimex/ingest/actions

## Manual Release (Alternative)

If you prefer manual releases:

```bash
# Create tag
git tag v1.0.1
git push origin v1.0.1

# Create release via GitHub UI
# The release workflow will auto-publish to PyPI
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backwards compatible
- **PATCH** (0.0.1): Bug fixes

## Workflows

### 1. auto-release.yml
- Triggers: Push to main with `setup.py` changes
- Creates tag and GitHub release automatically

### 2. release.yml
- Triggers: When GitHub release is published
- Builds and publishes package to PyPI

### 3. test.yml
- Triggers: Every push to main
- Runs tests on Python 3.10, 3.11, 3.12

## Requirements

- `PYPI_API_TOKEN` secret configured in repository settings ✅
- Valid PyPI account

## Troubleshooting

### Release Failed

Check the Actions tab:
- https://github.com/therealtimex/ingest/actions

Common issues:
- Version already exists on PyPI
- Invalid PYPI_API_TOKEN
- Build errors

### Skip Auto-Release

If you update `setup.py` but don't want to release:
- The tag will only be created if the version changed
- Existing tags are never recreated

## First Release Checklist

- [ ] Update version in `setup.py` to `1.0.0`
- [ ] Commit and push
- [ ] Verify workflows run successfully
- [ ] Check package on PyPI: https://pypi.org/project/ingest-cli/
- [ ] Test installation: `pip install ingest-cli`
