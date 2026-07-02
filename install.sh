#!/usr/bin/env bash
# leakit installer.
#
#   curl -fsSL https://raw.githubusercontent.com/victormaricato/leakit/main/install.sh | bash
#
# Installs the `leakit` CLI into an isolated environment using uv (preferred)
# or pipx. Override the source with LEAKIT_SOURCE, e.g.
#   LEAKIT_SOURCE="git+https://github.com/victormaricato/leakit.git" bash install.sh
set -euo pipefail

SOURCE="${LEAKIT_SOURCE:-leakit}"
GIT_FALLBACK="git+https://github.com/victormaricato/leakit.git"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then return 0; fi
  say "uv not found; installing it..."
  curl -fsSL https://astral.sh/uv/install.sh | sh
  # uv installs to ~/.local/bin by default
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1
}

install_with_uv() {
  say "Installing leakit from '$SOURCE' with uv..."
  if ! uv tool install --force "$SOURCE" 2>/dev/null; then
    warn "install from '$SOURCE' failed; falling back to GitHub source"
    uv tool install --force "$GIT_FALLBACK"
  fi
}

install_with_pipx() {
  say "Installing leakit from '$SOURCE' with pipx..."
  if ! pipx install --force "$SOURCE" 2>/dev/null; then
    warn "install from '$SOURCE' failed; falling back to GitHub source"
    pipx install --force "$GIT_FALLBACK"
  fi
}

main() {
  command -v curl >/dev/null 2>&1 || die "curl is required"
  if ensure_uv; then
    install_with_uv
  elif command -v pipx >/dev/null 2>&1; then
    install_with_pipx
  else
    die "need uv or pipx to install; see https://docs.astral.sh/uv/"
  fi

  echo
  say "leakit installed. Verify with:  leakit --version"
  cat <<'EOF'

Next steps:
  1. Export the API key for the service you want to probe, e.g.
       export LEAKIT_API_KEY="sk-..."          # or OPENAI_API_KEY
  2. Score a document:
       leakit --model gpt-4o-mini suspect.txt
  3. Probe any OpenAI-compatible endpoint with --base-url / --api-key-env.

If `leakit` is not found, add uv's tool bin to your PATH:
       export PATH="$HOME/.local/bin:$PATH"
EOF
}

main "$@"
