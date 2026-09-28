# SatNet — Satellite Close-Approach / Collision-Risk Evaluation System

SatNet ingests Two-Line Element (TLE) sets, propagates satellites with the SGP4
model, screens pairs of satellites for geometric close approaches, and reports
conjunction events with explicit scientific boundaries.

**SGP4 is the current physical propagation baseline.**

**Probability of collision requires appropriate covariance information. SatNet
does not fabricate covariance or Pc: without a covariance source, Pc is
reported as `unavailable`, never invented.**

## Current capabilities

- TLE / 3LE parsing, checksum + structural validation, SGP4 compatibility check
- Remote TLE fetching (CelesTrak by default) with injectable HTTP transport
- SGP4 propagation (via the established `sgp4` library — not re-implemented)
- Deterministic multi-satellite trajectories on a shared UTC time grid
- Geometric close-approach screening (cKDTree) with TCA refinement by
  re-propagating actual SGP4 states (golden-section search, no interpolation)
- Risk classification from miss-distance thresholds (or real Pc, if ever
  available)
- REST API (FastAPI), CSV / PDF reports, simulation persistence (SQLite /
  MySQL)
- React + Three.js dashboard plotting real TEME trajectories

## What SatNet does NOT do (by design)

- No probability of collision (Pc) without covariance — geometric miss
  distance is NOT Pc
- No maneuver recommendations of any kind
- No ML/DL accuracy claims; ML is an adapter boundary that can only fail back
  to the untouched SGP4 result (see `satnet/physics/ml_boundary.py`)
- No residual prediction-error calculations
- No frame conversion: SGP4 output stays in TEME unless an explicit,
  documented transformation is added

## Architecture

```text
React + Three.js (frontend/)          — talks to the API only
        │ REST JSON
        ▼
FastAPI (satnet/api)                   — HTTP boundary, no physics
        ▼
Application services (satnet/application)
        ├── SimulationService (orchestration)
        ├── SimulationStore (in-process + repository)
        └── TLEIngestionService (fetch → parse → validate → dedupe)
        ▼
Physics / Risk (satnet/physics, satnet/risk)
        ├── SGP4Propagator / TrajectoryPropagationService  (TEME, km, km/s)
        ├── SpatialScreeningService (cKDTree) + ConjunctionRefiner
        ├── RiskClassifier (thresholds)
        └── ProbabilityOfCollisionCalculator (Pc boundary: explicit N/A)
        ▼
Infrastructure (satnet/database)       — SQLAlchemy, isolated from physics
```

Layers never skip: the API never calls SGP4 directly, physics never imports
databases or HTTP, and the frontend never imports Python modules.

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"    # Windows
# .venv/bin/python -m pip install -e ".[dev]"          # Linux/macOS
```

## Running tests

```bash
.venv/Scripts/python.exe -m pytest
```

All network access in tests is mocked; the suite is fully offline and
deterministic (fixed TLE fixtures).

## Running examples

```bash
.venv/Scripts/python.exe examples/propagate_iss.py     # single-satellite SGP4
.venv/Scripts/python.exe examples/screen_satellites.py # two-satellite screening
```

Both use fixed TLE fixtures — no network required.

## Running the API + dashboard

```bash
.venv/Scripts/python.exe -m uvicorn satnet.api.main:app --reload
# frontend
cd frontend && npm install && npm run dev
```

Backend: http://localhost:8000 (docs at `/docs`) · Frontend: http://localhost:5173

The dashboard loads four offline SGP4 scenarios from
`frontend/public/demo/tle/`; each scenario contains four satellites and
includes close-orbit pairs for demonstrating screening. The model tab can
load `frontend/public/demo/model_demo.csv`. These are ordinary files, not
embedded in the React source, so they can be replaced with user data or fresh
TLEs.

## Coordinate frames, units, and time

| Quantity     | Value                                        |
|--------------|----------------------------------------------|
| Input        | TLE orbital elements (SGP4 interpretation)   |
| Output frame | **TEME** (True Equator, Mean Equinox) — native SGP4 frame, never silently converted |
| Position     | **km**                                       |
| Velocity     | **km/s**                                     |
| Time system  | **UTC** (timezone-aware timestamps enforced; naive datetimes rejected) |
| Gravity model| **WGS-72** (baked into the standard TLE/SGP4 model) |

## SGP4 limitations

- Accuracy degrades with time from the TLE epoch (typically a few good days
  for LEO); it is a predictive model, not truth.
- SGP4 does not model maneuvers.
- TLE element sets are only consistent with the WGS-72 SGP4 gravity model.
- Propagation errors (e.g. decayed objects) raise explicit
  `PropagationError`s rather than returning bad data.

## Screening vs. probability of collision

| Geometric screening (implemented)                  | Probabilistic conjunction assessment (NOT implemented) |
|----------------------------------------------------|--------------------------------------------------------|
| Relative position/velocity, separation distance    | Requires covariance of both objects                    |
| Minimum separation ("screening miss distance")     | Requires encounter-plane geometry + HBR                |
| Sampled grid + local SGP4 re-propagation refinement | Yields a probability (Pc)                              |

Sampled-time screening can miss fast encounters between grid points; the TCA
refiner only searches inside a bracketing interval of ±1 grid step around a
candidate. Reports label the miss distance as a screening value and show
`Pc: N/A (unavailable)` until a real covariance source and a validated
encounter-plane method exist.

## Configuration

Environment variables (see `.env.example`): `TLE_SOURCE_URL`,
`DEFAULT_SAFETY_RADIUS_KM`, `DEFAULT_TIME_STEP_SECONDS`,
`MAX_SIMULATION_HOURS`, `DATABASE_URL`, `ML_ENABLED`, `CORS_ORIGINS`.

## Future ML/DL architecture

```text
TLE / Orbit Data → SGP4 → Physics Baseline
                              ├─→ ML/DL Model → Experimental Result
                              └─→ Evaluation
```

SGP4 always remains the fallback: `SafeTrajectoryModel` returns the original
trajectories with an explicit warning if ML inference fails. No ML accuracy is
claimed anywhere in the codebase or docs, and no ML model ships until a real
training/validation experiment exists.

## Dependencies

Runtime: `sgp4`, `numpy`, `scipy`, `fastapi`, `uvicorn`, `pydantic`,
`pydantic-settings`, `httpx`, `sqlalchemy`, `pymysql`, `reportlab`,
`python-multipart`.
Dev: `pytest`, `pytest-cov`.

## Remaining work

- Covariance ingestion + validated encounter-plane Pc method (Pc stays
  `unavailable` until then)
- Optional analytical closest-approach refinement beyond the bracketed search
- ML research layer (experimental, behind the existing boundary)
- Frame transformations (ITRS/GCRS) if required — must be explicit and tested
- GPU-independent performance hardening for very large catalogs
