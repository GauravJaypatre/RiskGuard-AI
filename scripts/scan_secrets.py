"""Secret scanning script for RiskGuard AI repository.
Scans all files (excluding ignored build/cache directories) for credential patterns.
"""
import os
import re

PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "Google API Key (AIza...)"),
    (re.compile(r"AQ\.[A-Za-z0-9_\-]{20,}"), "Gemini API Key Pattern (AQ....)"),
    (re.compile(r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-\.]{10,}['\"]"), "Hardcoded Secret Assignment"),
    (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"), "Private Key"),
    (re.compile(r"ghp_[0-9a-zA-Z]{36}"), "GitHub Personal Access Token"),
    (re.compile(r"sk-[0-9a-zA-Z]{20,}"), "OpenAI / Generic Secret Key"),
    (re.compile(r"rzp_(?:test|live)_[0-9a-zA-Z]{14,}"), "Razorpay Key ID / Secret"),
]

SKIP_DIRS = {
    "node_modules", ".git", "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", "dist", "build"
}

def scan_repository(root_dir="."):
    findings = []
    scanned_count = 0
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            path = os.path.join(root, file)
            # Skip .env itself as we know it contains active secret and must be gitignored
            if file == ".env":
                continue
            # Skip files larger than 2MB (data files, binaries)
            try:
                if os.path.getsize(path) > 2 * 1024 * 1024:
                    continue
            except Exception:
                continue

            scanned_count += 1
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        for pattern, desc in PATTERNS:
                            for m in pattern.finditer(line):
                                matched = m.group(0)
                                preview = matched[:12] + "..." if len(matched) > 12 else matched
                                findings.append({
                                    "file": path,
                                    "line": line_num,
                                    "type": desc,
                                    "snippet": preview
                                })
            except Exception:
                pass
    return scanned_count, findings

if __name__ == "__main__":
    count, findings = scan_repository()
    print(f"Scanned {count} files across repository.")
    if not findings:
        print("RESULT: CLEAN! No hardcoded secrets or API keys found.")
    else:
        print(f"RESULT: Found {len(findings)} potential secret(s):")
        for item in findings:
            print(f"  - [{item['type']}] in {item['file']}:{item['line']} -> {item['snippet']}")
