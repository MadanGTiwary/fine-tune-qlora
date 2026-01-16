import json
import random
from pathlib import Path

# --- Configuration ---
NUM_SAMPLES = 50  # number of examples to generate
RAW_PATH = Path("data/raw/support_tickets.jsonl")
TRAIN_PATH = Path("data/processed/train.jsonl")

RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Data templates ---
issues = [
    "VPN not connecting on macOS",
    "Unable to access internal SharePoint site",
    "Email not syncing on mobile device",
    "Printer not found on office network",
    "Slow internet connection in remote office",
    "Docker build failing due to permission denied",
    "Kubernetes pod stuck in CrashLoopBackOff",
    "Jenkins build failing with missing dependencies",
    "AWS EC2 instance unreachable via SSH",
    "Git merge conflict on production branch",
    "Slack notifications not triggering for CI pipeline",
    "Outlook keeps prompting for password repeatedly",
    "Python script failing with 'ModuleNotFoundError'",
    "Windows update causing system restart loop",
    "Unable to access corporate VPN due to MFA issue",
    "Azure VM disk space running low",
    "Cloud storage bucket permission denied",
    "PostgreSQL database connection timeout",
    "Node.js server crashing due to memory leak",
    "Terraform apply failing with S3 backend error"
]

resolutions = [
    "Restart VPN service, reinstall client, and verify DNS settings.",
    "Check site permissions and group access in SharePoint admin center.",
    "Remove and re-add the email account; verify ActiveSync configuration.",
    "Reinstall printer driver and ensure network discovery is enabled.",
    "Check router settings and contact ISP to verify line stability.",
    "Ensure Docker daemon runs with correct privileges and update user group.",
    "Inspect pod logs, increase memory limits, and check container health probe.",
    "Update Jenkinsfile dependencies and clear cached workspaces.",
    "Verify EC2 security groups, key pair, and instance public IP.",
    "Use git mergetool or rebase from main branch to resolve conflicts.",
    "Verify Slack webhook URL and update CI/CD pipeline config.",
    "Clear credential cache in Outlook and reauthenticate with Microsoft 365.",
    "Install missing Python module via pip and check PYTHONPATH.",
    "Roll back recent Windows update or boot into Safe Mode for recovery.",
    "Re-register MFA device and reset VPN authentication token.",
    "Expand the Azure disk via portal and resize filesystem from CLI.",
    "Check IAM policy and grant proper object-level permissions.",
    "Verify DB host and port; increase connection timeout in config file.",
    "Profile memory usage using node --inspect and fix circular references.",
    "Check S3 bucket region match and verify backend access credentials."
]

# --- Generate synthetic dataset ---
def generate_dataset(num_samples: int = 50):
    data = []
    for i in range(num_samples):
        issue = random.choice(issues)
        resolution = random.choice(resolutions)
        entry = {
            "ticket_id": f"IT-{1000+i}",
            "issue": issue,
            "resolution": resolution
        }
        data.append(entry)
    return data


def write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- Main ---
if __name__ == "__main__":
    dataset = generate_dataset(NUM_SAMPLES)

    # Save raw dataset
    write_jsonl(RAW_PATH, dataset)
    print(f"✅ Saved raw dataset to {RAW_PATH} ({len(dataset)} records)")

    # Convert for train.jsonl (instruction-response format)
    train_dataset = [
        {
            "instruction": "You are an internal IT support assistant. Resolve the issue below.",
            "input": rec["issue"],
            "output": rec["resolution"]
        }
        for rec in dataset
    ]

    write_jsonl(TRAIN_PATH, train_dataset)
    print(f"✅ Saved training dataset to {TRAIN_PATH} ({len(train_dataset)} records)")
