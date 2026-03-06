#!/usr/bin/env bash
set -euo pipefail

# Security commit agent:
# - Scans staged content for likely secrets.
# - Enforces a small set of container/Kubernetes security guardrails.
# - Fails the commit on hard violations; warns on style issues.
#
# Targeted suppression (preferred over SECURITY_AGENT_BYPASS=1):
#   Per-line:  append  # security-agent: ignore  to the offending line
#   Per-file:  add the path/glob to .securityagentignore at the repo root

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "security-agent: not in a git repository"
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

if [[ "${SECURITY_AGENT_BYPASS:-}" == "1" ]]; then
  echo "security-agent: bypass enabled via SECURITY_AGENT_BYPASS=1"
  exit 0
fi

mapfile -t staged_files < <(git diff --cached --name-only --diff-filter=ACMR)
if [[ "${#staged_files[@]}" -eq 0 ]]; then
  exit 0
fi

# Load .securityagentignore patterns (gitignore-style: globs, # comments)
ignored_patterns=()
if [[ -f "${repo_root}/.securityagentignore" ]]; then
  while IFS= read -r line; do
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    ignored_patterns+=("${line}")
  done < "${repo_root}/.securityagentignore"
fi

is_ignored_file() {
  local file="$1"
  for pattern in "${ignored_patterns[@]}"; do
    # shellcheck disable=SC2254
    if [[ "${file}" == ${pattern} ]]; then
      return 0
    fi
  done
  return 1
}

failures=()
warnings=()

add_failure() { failures+=("$1"); }
add_warning()  { warnings+=("$1"); }

contains_suspicious_secret() {
  local input="$1"

  # Hard secret signatures.
  if grep -Eq -- 'AKIA[0-9A-Z]{16}' <<<"${input}"; then
    return 0
  fi
  if grep -Eq -- '-----BEGIN (RSA|EC|OPENSSH|DSA|PGP) PRIVATE KEY-----' <<<"${input}"; then
    return 0
  fi

  # Generic key/token/password assignment with high-entropy-ish value.
  if grep -Eqi -- '(api[_-]?key|secret|token|password)[[:space:]]*[:=][[:space:]]*["'"'"']?[A-Za-z0-9_./+=-]{12,}' <<<"${input}"; then
    # Skip obvious templates/placeholders.
    if grep -Eqi -- '(\$\{|{{|<change-me>|changeme|example|redacted|template)' <<<"${input}"; then
      return 1
    fi
    return 0
  fi

  return 1
}

risky_config_reasons() {
  local input="$1"
  local reasons=()

  if grep -Eqi '(^|[[:space:]])privileged:[[:space:]]*true([[:space:]]|$)' <<<"${input}"; then
    reasons+=("BLOCK:privileged: true grants near-host-level privileges and can allow container breakout.")
  fi
  if grep -Eqi '(^|[[:space:]])allowPrivilegeEscalation:[[:space:]]*true([[:space:]]|$)' <<<"${input}"; then
    reasons+=("BLOCK:allowPrivilegeEscalation: true lets processes gain extra privileges (for example via setuid binaries).")
  fi
  if grep -Eqi '(^|[[:space:]])runAsNonRoot:[[:space:]]*false([[:space:]]|$)' <<<"${input}"; then
    reasons+=("BLOCK:runAsNonRoot: false allows root in-container execution, which increases blast radius.")
  fi
  if grep -Eqi 'curl[[:space:]].*[|][[:space:]]*(sh|bash)' <<<"${input}"; then
    reasons+=("BLOCK:curl-pipe-sh executes remote code without integrity pinning or review.")
  fi
  # :latest is a style warning, not a hard block
  if grep -Eqi '(^|[[:space:]])image:[[:space:]]*[^[:space:]]+:latest([[:space:]]|$)' <<<"${input}"; then
    reasons+=("WARN:image tag :latest is mutable and breaks reproducibility/supply-chain traceability.")
  fi

  if [[ "${#reasons[@]}" -eq 0 ]]; then
    return 1
  fi

  printf '%s\n' "${reasons[@]}"
  return 0
}

for file in "${staged_files[@]}"; do
  # Block committing known local secret artifacts.
  if [[ "${file}" =~ (^|/)(1password-credentials\.json|secrets\.env\.json|\.env(\.|$)|id_rsa|id_ed25519) ]]; then
    add_failure "blocked sensitive file path in staged changes: ${file}"
    continue
  fi

  if is_ignored_file "${file}"; then
    echo "security-agent: skipping ignored file: ${file}"
    continue
  fi

  # Added lines only, stripping per-line suppressions.
  added_lines="$(git diff --cached -U0 -- "${file}" | sed -n 's/^+//p' | sed '/^+++/d' | grep -v '# security-agent: ignore' || true)"
  [[ -z "${added_lines}" ]] && continue

  # For secret scanning, exclude lines containing URL-embedded credentials
  # (e.g. rtsp://user:pass@host, https://token@host) — these are not plaintext leaks.
  secret_scan_lines="$(grep -Ev '[a-z]+://[^[:space:]]*' <<<"${added_lines}" || true)"
  if contains_suspicious_secret "${secret_scan_lines}"; then
    add_failure "potential secret detected in staged additions: ${file}"
  fi

  if risky_output="$(risky_config_reasons "${added_lines}")"; then
    while IFS= read -r reason; do
      [[ -z "${reason}" ]] && continue
      if [[ "${reason}" == BLOCK:* ]]; then
        add_failure "risky security pattern in ${file}: ${reason#BLOCK:}"
      elif [[ "${reason}" == WARN:* ]]; then
        add_warning "risky security pattern in ${file}: ${reason#WARN:}"
      fi
    done <<<"${risky_output}"
  fi
done

if [[ "${#warnings[@]}" -gt 0 ]]; then
  echo "security-agent: warnings"
  printf ' - %s\n' "${warnings[@]}"
fi

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "security-agent: commit blocked"
  printf ' - %s\n' "${failures[@]}"
  echo ""
  echo "Targeted suppression options:"
  echo "  Per-line: append '# security-agent: ignore' to the offending line"
  echo "  Per-file: add the path or glob to .securityagentignore"
  exit 1
fi

echo "security-agent: passed"
