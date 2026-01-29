FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	UV_BREAK_SYSTEM_PACKAGES=true \
	TZ=Asia/Bangkok

RUN ln -sf /usr/share/zoneinfo/Asia/Bangkok /etc/localtime && echo "Asia/Bangkok" > /etc/timezone

WORKDIR /app

# Install uv (Rust binary distributed via PyPI)
RUN python -m pip install --no-cache-dir uv

# Copy dependency metadata first (better layer caching)
# Note: context is parent directory, so paths are relative to project root
COPY ../pyproject.toml ../uv.lock ./

# Export locked deps (include your optional extra that depends on torch)
# IMPORTANT: do NOT enable the "torch" extra if you don't want uv to pull torch wheels.
RUN uv export --frozen --extra depend-torch -o /tmp/requirements.txt 

# Install into the SYSTEM Python environment (same env that already has torch from base image)
RUN uv pip install --system -r /tmp/requirements.txt 

# Copy the rest of your source code
COPY . .

WORKDIR /app/SkateFormer
CMD ["python", "main.py", "--help"]
