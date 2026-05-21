#!/bin/bash
set -e

# Always operate from the repo root regardless of where the caller invokes us.
cd "$(dirname "$0")/.."

# Add bun to PATH
export PATH="/home/amaterasu/.bun/bin:$PATH"

PROJECT_ID="nexus-cluster"

echo "🎛️  Nexus Deployment"
echo "--------------------"

echo "[*] Installing + building..."
bun install
bun run build

echo "[*] Deploying to Firebase Hosting..."
if command -v firebase &> /dev/null; then
    firebase deploy --only hosting --project $PROJECT_ID
else
    echo "[*] firebase command not found, using bunx..."
    bunx firebase-tools deploy --only hosting --project $PROJECT_ID
fi

echo "✅ Nexus deployed: https://$PROJECT_ID.web.app"
