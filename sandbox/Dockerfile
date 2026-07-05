# Hardened sandbox base image for agent-generated quant research scripts.
# A real quant research stack, baked in so the offline sandbox has professional
# tools by default. Agent-declared extra packages are layered on top at
# provision time (see app/quant/provision.py) — the research code itself never
# has network access.
FROM python:3.12-slim

# Build tooling for any source-only deps (cvxpy solvers, arch), removed after.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "numpy>=1.26" "pandas>=2.2" "scipy>=1.13" \
        "scikit-learn>=1.5" "statsmodels>=0.14" \
        "empyrical-reloaded>=0.5" "alphalens-reloaded>=0.4" \
        "arch>=7" "cvxpy>=1.5" "quantstats>=0.0.6" \
        "matplotlib>=3.8" "plotly>=5.20" \
    && useradd --create-home --uid 10001 sandbox \
    && mkdir -p /out && chown sandbox /out

WORKDIR /app
COPY runner.py /app/runner.py

ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl ARTIFACTS_DIR=/out
USER sandbox

# Usage: python /app/runner.py /in/research.py <seed>
ENTRYPOINT ["python", "/app/runner.py"]
