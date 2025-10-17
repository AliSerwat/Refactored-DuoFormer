import os, requests

token = os.getenv("GITHUB_TOKEN")
if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable not set")

owner = "AliSerwat"
repo = "Refactored-DuoFormer"
run_ids = [18594083765, 18594083737, 18594083726]

headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

for run_id in run_ids:
    print(f"\n=== Run {run_id} ===")
    jobs_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
    jobs_resp = requests.get(jobs_url, headers=headers)
    jobs_resp.raise_for_status()
    jobs = jobs_resp.json()["jobs"]

    for job in jobs:
        print(f"Job: {job['name']} — {job['status']}/{job['conclusion']}")
        # Fetch detailed steps for each job
        for step in job.get("steps", []):
            name = step.get("name", "<unnamed>")
            status = step.get("status")
            concl = step.get("conclusion")
            print(f"  • {name}: {status}/{concl}")

        # Optionally download full logs ZIP:
        # logs_url = job["logs_url"]
        # logs_resp = requests.get(logs_url, headers=headers)
        # with open(f"run_{run_id}_job_{job['id']}_logs.zip", "wb") as f:
        #     f.write(logs_resp.content)
