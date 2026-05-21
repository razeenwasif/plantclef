#!/bin/bash
set -e

# Always operate from the repo root regardless of where the caller invokes us.
cd "$(dirname "$0")/.."

# Add bun to PATH
export PATH="/home/amaterasu/.bun/bin:$PATH"

PROJECT_ID="oracle-neuro-sym"

echo "🎨 STAGE: Frontend Deployment (Firebase Hosting) using Bun"
echo "----------------------------------------------------"

# Sync the latest paper sources into the dashboard's public folder so the
# Research tab can serve a fresh PDF download and Copy-LaTeX action.
# Currently syncs the legacy paper/ folder (what the dashboard's Research
# tab content describes). The official PlantCLEF2026-main/report/main.pdf
# is intentionally NOT auto-served here — it's stashed at /tmp by the
# operator until the dashboard tab is updated to match the new paper.
if [ -f paper/plantclef2026_research_paper.tex ]; then
    cp paper/plantclef2026_research_paper.tex dashboard/public/oracle_plantclef2026.tex
    echo "[*] Synced paper .tex (legacy)"
fi
if [ -f paper/plantclef2026_research_paper.pdf ]; then
    cp paper/plantclef2026_research_paper.pdf dashboard/public/oracle_plantclef2026.pdf
    echo "[*] Synced paper PDF (legacy)"
fi

# Build the React dashboard
echo "[*] Building dashboard with Bun..."
cd dashboard
bun install
bun run build
cd ..

# Deploy to Firebase
echo "[*] Deploying to Firebase Hosting..."
# Use npx if firebase isn't in path, but try to use local firebase-tools if possible
if command -v firebase &> /dev/null; then
    firebase deploy --only hosting --project $PROJECT_ID
else
    echo "[*] firebase command not found, using npx..."
    npx firebase-tools deploy --only hosting --project $PROJECT_ID
fi

echo "✅ Frontend deployed successfully: https://$PROJECT_ID.web.app"
