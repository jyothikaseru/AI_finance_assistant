import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["@xenova/transformers", "onnxruntime-node"],
  allowedDevOrigins: ["172.23.43.11", "localhost:3000"],
};

export default nextConfig;
