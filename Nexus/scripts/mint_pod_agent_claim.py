#!/usr/bin/env python3
"""
Mint the `pod_agent` + `pod_id` custom claims on a Firebase service account.

Run **once per pod** on a trusted operator machine (not the pod itself).
This is the boundary between "anyone with a service-account JSON" and "this
specific JSON can only write /pods/{pod_id}". See `firestore.rules` and
`docs/CONTROL_PLANE.md` (Bootstrapping a pod) for context.

Usage
-----
.. code-block:: bash

    python scripts/mint_pod_agent_claim.py \\
        --admin-credentials /path/to/firebase-admin-sdk.json \\
        --service-account-uid <UID of the service-account user> \\
        --pod-id pod-5090

The ``--service-account-uid`` is the UID of the Firebase *user* the
service-account JSON authenticates as — *not* the GCP service account
resource id. The easiest way to grab it is to sign in once with the
service account and check ``auth.currentUser.uid``; in practice you
mint it via the Admin SDK at the same time you create the user.

This script is intentionally short — it's a wrapper over
``firebase_admin.auth.set_custom_user_claims``.
"""

from __future__ import annotations

import argparse
import sys

try:
    import firebase_admin
    from firebase_admin import auth as fb_auth, credentials
except ImportError:
    sys.exit("firebase-admin is required. `pip install firebase-admin`")


def main() -> int:
    p = argparse.ArgumentParser(description="Mint pod_agent + pod_id custom claims.")
    p.add_argument("--admin-credentials", required=True,
                   help="Path to a *Firebase Admin SDK* service-account JSON "
                        "(distinct from the pod-agent service account being claimed).")
    p.add_argument("--service-account-uid", required=True,
                   help="UID of the pod-agent service-account user.")
    p.add_argument("--pod-id", required=True,
                   help="Pod id the agent is allowed to write (must match cluster.yaml).")
    args = p.parse_args()

    cred = credentials.Certificate(args.admin_credentials)
    firebase_admin.initialize_app(cred)

    claims = {"pod_agent": True, "pod_id": args.pod_id}
    fb_auth.set_custom_user_claims(args.service_account_uid, claims)
    print(f"OK  → uid={args.service_account_uid}  claims={claims}")
    print("Note: the pod agent must re-authenticate (restart) to pick up new claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
