#!/usr/bin/env bash
set -euo pipefail

# Function to ensure cargo-edit is installed
ensure_cargo_edit() {
    if ! cargo set-version --help >/dev/null 2>&1; then
        echo "Installing cargo-edit..."
        cargo install cargo-edit
    fi
}

# Check for version bump type argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <major|minor|patch|version>"
    echo "Examples:"
    echo "  $0 major     # Bumps major version"
    echo "  $0 minor     # Bumps minor version"
    echo "  $0 patch     # Bumps patch version"
    echo "  $0 1.2.3     # Sets specific version"
    exit 1
fi

VERSION_ARG="$1"

# Ensure we have cargo-edit
ensure_cargo_edit

# Get the current version from Cargo.toml
CURRENT_VERSION=$(cargo metadata --format-version 1 --no-deps | jq -r '.packages[0].version')

# Handle version bump or set
if [[ "$VERSION_ARG" =~ ^(major|minor|patch)$ ]]; then
    # Bump version using cargo-edit
    cargo set-version --bump "$VERSION_ARG"
    NEW_VERSION=$(cargo metadata --format-version 1 --no-deps | jq -r '.packages[0].version')
elif [[ "$VERSION_ARG" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    # Set specific version using cargo-edit
    cargo set-version "$VERSION_ARG"
    NEW_VERSION="$VERSION_ARG"
else
    echo "Error: Invalid version argument. Must be 'major', 'minor', 'patch', or a semver version (e.g., '1.2.3')"
    exit 1
fi

# Update pyproject.toml using sed but with more precise TOML awareness
echo "Updating pyproject.toml..."
sed -i.bak "/^version[[:space:]]*=[[:space:]]*.*$/c\version = \"$NEW_VERSION\"" fish_speech_python/pyproject.toml
rm fish_speech_python/pyproject.toml.bak

# Create git tag
echo "Creating git tag v$NEW_VERSION..."
git add Cargo.toml Cargo.lock fish_speech_python/pyproject.toml
git commit -m "chore: bump version from $CURRENT_VERSION to $NEW_VERSION"
# Not doing this, create tag manually
# git tag -a "v$NEW_VERSION" -m "Version $NEW_VERSION"

echo "Done! Changes committed."
echo ""
echo "Next steps:"
echo "1. Review the changes: git show HEAD"
echo "2. Push the changes: git push origin main"
echo "3. Push the tag: git push origin v$NEW_VERSION"
