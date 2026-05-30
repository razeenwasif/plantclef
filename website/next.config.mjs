/** @type {import('next').NextConfig} */

// We host on arm-wision.github.io (an org root site), so the app is served
// from the domain root, no basePath needed. The BASE_PATH env var stays as
// an escape hatch in case we ever deploy under a subpath again (set it to
// e.g. "/PlantCLEF2026" in that case).
const basePath = process.env.BASE_PATH || "";

const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,

  basePath:    basePath || undefined,
  assetPrefix: basePath ? basePath + "/" : undefined,

  // Exposed so client code (lib/paths.asset) can prefix raw <img src> values.
  // When basePath is empty the helper is a no-op.
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },

  reactStrictMode: true,
};

export default nextConfig;
