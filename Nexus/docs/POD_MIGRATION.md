# Pod migration · Oracle → Nexus

One-pager for moving training pods from the original shared-codebase
Firebase project (`oracle-neuro-sym`) to the standalone Nexus project
(`nexus-cluster`). Run this once per training host.

## What you'll need

- SSH access to the training host
- A fresh service-account JSON key for the `nexus-cluster` Firebase
  project (Firebase Console → **Project Settings** → **Service Accounts**
  → **Generate new private key**)
- The Firebase UID of the user the pod agent authenticates as. If you
  don't have a dedicated bot user, create one in
  **Authentication → Users → Add user**, then copy its UID from the
  Users table.

## Per-pod migration recipe

```bash
# 1. Stop the old Oracle-era unit (still authenticating against
#    oracle-neuro-sym; the /pods doc will go offline cleanly).
sudo systemctl stop oracle-pod-agent
sudo systemctl disable oracle-pod-agent

# 2. Lay out the Nexus install directory.
sudo mkdir -p /opt/nexus/scripts /opt/nexus/configs /etc/nexus
sudo useradd --system --home /opt/nexus --shell /bin/bash nexus  # if not already present
sudo chown -R nexus:nexus /opt/nexus

# 3. Copy the agent code + cluster manifest.
sudo cp ~/Nexus/scripts/pod_agent.py        /opt/nexus/scripts/
sudo cp ~/Nexus/scripts/run_monitor.py      /opt/nexus/scripts/
sudo cp ~/Nexus/scripts/run_spawner.py      /opt/nexus/scripts/
sudo cp /path/to/cluster.yaml               /opt/nexus/configs/

# 4. Drop the new service-account JSON in /etc/nexus and lock it down.
sudo cp ~/nexus-credentials.json /etc/nexus/pod_agent.json
sudo chown root:nexus /etc/nexus/pod_agent.json
sudo chmod 640         /etc/nexus/pod_agent.json
```

## Mint the agent's custom claims (one-time, from any workstation)

The Firestore rules require the pod agent's auth token to carry
`pod_agent: true` and `pod_id: <this-pod's-id>` custom claims. Mint them
with the helper script:

```bash
python3 ~/Nexus/scripts/mint_pod_agent_claim.py \
    --credentials /path/to/nexus-credentials.json \
    --uid <firebase-user-uid-from-console> \
    --pod-id pod-5090
```

The script uses the Admin SDK to call `auth.set_custom_user_claims`. On
the agent side, the next ID-token refresh (≤1 h, force a refresh on
agent start) picks up the new claims, after which writes to
`/pods/pod-5090` are accepted.

## Wire up the systemd unit

```bash
sudo cp ~/Nexus/scripts/nexus-pod-agent.service /etc/systemd/system/

# Edit POD_ID + paths if this host's setup differs from the defaults
# (Environment=POD_ID=..., WorkingDirectory=..., the ExecStart paths).
sudoedit /etc/systemd/system/nexus-pod-agent.service

sudo systemctl daemon-reload
sudo systemctl enable --now nexus-pod-agent
sudo journalctl -u nexus-pod-agent -f          # watch it come up
```

## Verify

1. Open https://nexus-cluster.web.app → **Fleet** tab.
2. Within ~30 seconds the pod should appear with `status=online`, the
   correct `display_name`, accelerator info, and a fresh
   `last_heartbeat`.
3. If it doesn't appear, `journalctl -u nexus-pod-agent` on the host
   surfaces the usual failure modes:
   - `PERMISSION_DENIED on /pods/<id>` → claim missing or wrong
     `pod_id`; re-run `mint_pod_agent_claim.py`, then restart the unit
     (force a token refresh)
   - `Could not find project` → wrong project ID; check
     `GOOGLE_APPLICATION_CREDENTIALS` points at the
     `nexus-cluster.json`, not the old Oracle one

## Optional cleanup

Once the new unit is happily heartbeating, you can remove the legacy
Oracle install from the host:

```bash
sudo rm -rf /opt/oracle /etc/oracle
sudo userdel oracle           # only if no other services run as `oracle`
sudo rm /etc/systemd/system/oracle-pod-agent.service
sudo systemctl daemon-reload
```

Or leave it in place if you want a documented rollback path. The agent
binary itself is tiny; only the service-account key is worth deleting
on security grounds.
