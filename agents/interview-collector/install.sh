#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install Interview Collector Agent templates.

Usage:
  bash agents/interview-collector/install.sh [--targets copilot,claude,cursor,generic] [--force]

Targets:
  copilot  Install Copilot CLI user instruction
  claude   Install Claude Code skill
  cursor   Install Cursor project rule into this repository
  generic  Install a generic prompt under ~/.agent-interview-hub
  all      Install all targets (default)

Options:
  --force  Overwrite existing files after creating timestamped .bak copies
EOF
}

TARGETS="all"
FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --targets)
      TARGETS="${2:-}"
      shift 2
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/templates"

contains_target() {
  local wanted="$1"
  [[ "$TARGETS" == "all" ]] && return 0
  [[ ",$TARGETS," == *",$wanted,"* ]]
}

copy_template() {
  local src="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"

  if [[ -e "$dest" ]]; then
    if cmp -s "$src" "$dest"; then
      echo "Already up to date: $dest"
      return
    fi

    if [[ "$FORCE" != "true" ]]; then
      echo "Refusing to overwrite existing file: $dest" >&2
      echo "Re-run with --force to create a .bak copy and overwrite it." >&2
      exit 1
    fi

    local backup="$dest.bak.$(date +%Y%m%d%H%M%S)"
    cp "$dest" "$backup"
    echo "Backed up existing file: $backup"
  fi

  cp "$src" "$dest"
  echo "Installed: $dest"
}

install_copilot() {
  local dest="$HOME/.copilot/instructions/interview-collector.instructions.md"
  copy_template "$TEMPLATE_DIR/copilot/interview-collector.instructions.md" "$dest"
}

install_claude() {
  local dest_dir="$HOME/.claude/skills/interview-collector"
  copy_template "$TEMPLATE_DIR/claude/SKILL.md" "$dest_dir/SKILL.md"
  copy_template "$SCRIPT_DIR/AGENT.md" "$dest_dir/AGENT.md"
  echo "Installed Claude skill: $dest_dir"
}

install_cursor() {
  local dest="$REPO_ROOT/.cursor/rules/interview-collector.mdc"
  copy_template "$TEMPLATE_DIR/cursor/interview-collector.mdc" "$dest"
}

install_generic() {
  local dest_dir="$HOME/.agent-interview-hub/interview-collector"
  copy_template "$TEMPLATE_DIR/generic/AGENT.md" "$dest_dir/AGENT.md"
  copy_template "$SCRIPT_DIR/AGENT.md" "$dest_dir/AGENT.full.md"
  echo "Installed generic prompt: $dest_dir"
}

if contains_target copilot; then install_copilot; fi
if contains_target claude; then install_claude; fi
if contains_target cursor; then install_cursor; fi
if contains_target generic; then install_generic; fi

echo
echo "Optional tool health check:"
if command -v agent-reach >/dev/null 2>&1; then
  agent-reach doctor --json >/dev/null 2>&1 && echo "  agent-reach: available" || echo "  agent-reach: installed but doctor reported issues"
else
  echo "  agent-reach: not installed (optional, recommended for web collection)"
fi

if command -v mcporter >/dev/null 2>&1; then
  echo "  mcporter: available"
else
  echo "  mcporter: not installed (optional, recommended for Exa search)"
fi

echo
echo "Done."
