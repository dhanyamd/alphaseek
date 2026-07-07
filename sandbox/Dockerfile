# Hardened sandbox base image for agent-generated quant research scripts.
# A real quant research stack, baked in so the offline sandbox has professional
# tools by default. Agent-declared extra packages are layered on top at
# provision time (see app/quant/provision.py) — the research code itself never
# has network access.
FROM python:3.12-slim

# Build tooling for any source-only deps (cvxpy solvers, arch), removed after.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Core scientific stack
RUN pip install --no-cache-dir \
        "numpy>=1.26" "pandas>=2.2" "scipy>=1.13" \
        "scikit-learn>=1.5" "statsmodels>=0.14" \
        "matplotlib>=3.8" "plotly>=5.20" "seaborn>=0.13" \
        "numba" "patsy" "arch>=7" "cvxpy>=1.5" "quantstats>=0.0.6" \
        "hmmlearn>=0.3" "yfinance" \
    && useradd --create-home --uid 10001 sandbox \
    && mkdir -p /out && chown sandbox /out

# Quant research stack (factor analysis, performance, portfolio optimization)
RUN pip install --no-cache-dir \
        "alphalens-reloaded" "empyrical-reloaded" \
        "cvxportfolio" "skfolio" "riskfolio-lib" "pyportfolioopt" "vectorbt" \
    || pip install --no-cache-dir \
        "alphalens-reloaded" "empyrical-reloaded" \
        "cvxportfolio" "skfolio" "riskfolio-lib" "pyportfolioopt" "vectorbt"

USER sandbox

WORKDIR /app
COPY runner.py /app/runner.py

ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl ARTIFACTS_DIR=/out
USER sandbox

# Usage: python /app/runner.py /in/research.py <seed>
ENTRYPOINT ["python", "/app/runner.py"]
