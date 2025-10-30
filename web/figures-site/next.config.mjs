/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/ferromic',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
