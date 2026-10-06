# Two images from one build: the static files alone, or the files behind nginx.
#
#   docker buildx build --target static ...   ->  FROM scratch + /dist        (the deploy uploads it to S3)
#   docker buildx build ...                   ->  nginx serving /dist on 8080 (local or container hosting)
#
# The production path is `static`: CloudFront serves the files from S3, so no
# web server runs in AWS. webapp/deploy/deploy.py --frontend-image pulls this
# image's one layer over HTTPS and uploads dist/ — no Docker, no Node.
#
# WHAT THIS DOES NOT DO
#   It bakes in no settings: the app calls /api on its own origin, so one
#   build works behind every student's CloudFront distribution.

# STEP 1  build the static app with Node. BUILDPLATFORM: npm runs natively on
#         the build machine whatever --platform asks for — the output is plain
#         HTML/JS/CSS, the same bytes on every platform.
FROM --platform=$BUILDPLATFORM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json .npmrc ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

# STEP 2a  the files and nothing else — what --publish-frontend pushes
FROM scratch AS static
COPY --from=build /app/dist /dist

# STEP 2b  (default target) serve them with nginx, as a non-root user on port 8080
FROM nginxinc/nginx-unprivileged:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
