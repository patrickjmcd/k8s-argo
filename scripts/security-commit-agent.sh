#!/usr/bin/env bash
set -euo pipefail

# Security commit agent:
# - Scans staged content for likely secrets.
# - Enforces a small set of container/Kubernetes security guardrails.
# - Fails the commit on violations.

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

failures=()

add_failure() {
  failures+=("$1")
}

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
    if grep -Eqi -- '(\\$\\{|{{|<change-me>|changeme|example|redacted|template)' <<<"${input}"; then
      return 1
    fi
    return 0
  fi

  return 1
}

risky_config_reasons() {
  local input="$1"
  local reasons=()

  # Guardrails focused on high-risk misconfigurations.
  if grep -Eqi '(^|[[:space:]])privileged:[[:space:]]*true([[:space:]]|$)' <<<"${input}"; then
    reasons+=("privileged: true grants near-host-level privileges and can allow container breakout.")
  fi
  if grep -Eqi '(^|[[:space:]])allowPrivilegeEscalation:[[:space:]]*true([[:space:]]|$)' <<<"${input}"; then
    reasons+=("allowPrivilegeEscalation: true lets processes gain extra privileges (for example via setuid binaries).")
  fi
  if grep -Eqi '(^|[[:space:]])runAsNonRoot:[[:space:]]*false([[:space:]]|$)' <<<"${input}"; then
    reasons+=("runAsNonRoot: false allows root in-container execution, which increases blast radius.")
  fi
  if grep -Eqi '(^|[[:space:]])image:[[:space:]]*[^[:space:]]+:latest([[:space:]]|$)' <<<"${input}"; then
    reasons+=("image tag :latest is mutable and breaks reproducibility/supply-chain traceability.")
  fi
  if grep -Eqi 'curl[[:space:]].*[|][[:space:]]*(sh|bash)' <<<"${input}"; then
    reasons+=("curl | sh executes remote code without integrity pinning or review.")
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
  fi

  # Added lines only.
  added_lines="$(git diff --cached -U0 -- "${file}" | sed -n 's/^+//p' | sed '/^+++/d')"
  [[ -z "${added_lines}" ]] && continue

  if contains_suspicious_secret "${added_lines}"; then
    add_failure "potential secret detected in staged additions: ${file}"
  fi

  if risky_output="$(risky_config_reasons "${added_lines}")"; then
    while IFS= read -r reason; do
      [[ -z "${reason}" ]] && continue
      add_failure "risky security pattern in ${file}: ${reason}"
    done <<<"${risky_output}"
  fi
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "security-agent: commit blocked"
  printf ' - %s\n' "${failures[@]}"
  echo "Set SECURITY_AGENT_BYPASS=1 to bypass for an exceptional one-off commit."
  exit 1
fi

echo "security-agent: passed"
