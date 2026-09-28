"""Hybrid risk assessment: fuse physics-based Pc with ML-based risk prediction.

This module implements the fusion engine that combines:
1. Physics-based Pc (Alfano/Foster encounter-plane integration) -- when
   covariance data is available.
2. ML-based risk (trained XGBoost model on 13,000+ real CDM events) -- always
   available when CDM data is supplied.

When both sources agree, the system reports high confidence.
When they disagree, it flags the event for human review.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from satnet.domain.models import RiskLevel


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class HybridRiskResult:
    """Combined risk assessment from physics + ML."""
    # Final fused assessment
    risk_level: RiskLevel
    confidence: ConfidenceLevel
    reason: str

    # Physics side (may be None if no covariance)
    physics_pc: float | None
    physics_risk_level: RiskLevel | None
    physics_reason: str | None

    # ML side (may be None if no CDM data)
    ml_pc: float | None
    ml_risk_level: str | None
    ml_reason: str | None

    # Meta
    needs_human_review: bool
    fusion_method: str


class HybridRiskEngine:
    """Fuses physics-based and ML-based risk assessments.

    Decision matrix:
        Physics RED   + ML RED    -> RED,    HIGH confidence
        Physics GREEN + ML GREEN  -> GREEN,  HIGH confidence
        Physics RED   + ML GREEN  -> YELLOW, flag for review
        Physics GREEN + ML RED    -> YELLOW, flag for review
        Physics N/A   + ML RED    -> RED,    MEDIUM confidence (ML-only)
        Physics RED   + ML N/A    -> RED,    MEDIUM confidence (physics-only)
    """

    # Thresholds matching the existing RiskClassifier
    RED_PC = 1e-4
    YELLOW_PC = 1e-6

    def fuse(
        self,
        miss_distance_km: float,
        physics_pc: float | None,
        ml_pc: float | None,
        ml_risk_level: str | None = None,
        ml_reason: str | None = None,
    ) -> HybridRiskResult:
        """Combine physics and ML risk signals into a single assessment."""

        # --- Classify physics side ---
        physics_level, physics_reason = self._classify_physics(miss_distance_km, physics_pc)

        # --- Classify ML side ---
        if ml_pc is not None and ml_risk_level is not None:
            ml_level = self._str_to_risk_level(ml_risk_level)
        elif ml_pc is not None:
            ml_level = self._classify_ml_pc(ml_pc)
            ml_reason = f"ML Pc = {ml_pc:.2e}"
        else:
            ml_level = None

        # --- Fusion logic ---
        if ml_level is not None and physics_pc is not None:
            # Both available -> highest confidence fusion
            return self._fuse_both(
                physics_level, physics_pc, physics_reason,
                ml_level, ml_pc, ml_reason,
                miss_distance_km,
            )
        elif ml_level is not None:
            # ML only (no covariance for physics)
            return HybridRiskResult(
                risk_level=ml_level,
                confidence=ConfidenceLevel.MEDIUM,
                reason=f"ML-only assessment (no covariance for physics Pc). {ml_reason}",
                physics_pc=None,
                physics_risk_level=physics_level,
                physics_reason=physics_reason,
                ml_pc=ml_pc,
                ml_risk_level=ml_risk_level or ml_level.value,
                ml_reason=ml_reason,
                needs_human_review=ml_level == RiskLevel.RED,
                fusion_method="ml_only",
            )
        elif physics_pc is not None:
            # Physics only (no CDM data for ML)
            return HybridRiskResult(
                risk_level=physics_level,
                confidence=ConfidenceLevel.MEDIUM,
                reason=f"Physics-only assessment (no CDM data for ML). {physics_reason}",
                physics_pc=physics_pc,
                physics_risk_level=physics_level,
                physics_reason=physics_reason,
                ml_pc=None,
                ml_risk_level=None,
                ml_reason=None,
                needs_human_review=physics_level == RiskLevel.RED,
                fusion_method="physics_only",
            )
        else:
            # Neither available -> geometric only
            return HybridRiskResult(
                risk_level=physics_level,
                confidence=ConfidenceLevel.LOW,
                reason=f"Geometric screening only (no Pc, no ML). {physics_reason}",
                physics_pc=None,
                physics_risk_level=physics_level,
                physics_reason=physics_reason,
                ml_pc=None,
                ml_risk_level=None,
                ml_reason=None,
                needs_human_review=False,
                fusion_method="geometric_only",
            )

    def _fuse_both(
        self,
        physics_level: RiskLevel,
        physics_pc: float,
        physics_reason: str,
        ml_level: RiskLevel,
        ml_pc: float | None,
        ml_reason: str | None,
        miss_distance_km: float,
    ) -> HybridRiskResult:
        """Fuse when both physics and ML assessments are available."""

        if physics_level == ml_level:
            # Agreement -> high confidence
            return HybridRiskResult(
                risk_level=physics_level,
                confidence=ConfidenceLevel.HIGH,
                reason=f"Physics and ML agree: {physics_level.value}. {physics_reason}",
                physics_pc=physics_pc,
                physics_risk_level=physics_level,
                physics_reason=physics_reason,
                ml_pc=ml_pc,
                ml_risk_level=ml_level.value if ml_level else None,
                ml_reason=ml_reason,
                needs_human_review=False,
                fusion_method="consensus",
            )

        # Disagreement -> take the more conservative (higher risk) and flag
        if self._risk_order(physics_level) < self._risk_order(ml_level):
            # Physics says higher risk
            final_level = physics_level
            reason = (
                f"DISAGREEMENT: Physics={physics_level.value} vs ML={ml_level.value}. "
                f"Using conservative (physics) assessment. Flagged for human review."
            )
        else:
            # ML says higher risk
            final_level = ml_level
            reason = (
                f"DISAGREEMENT: Physics={physics_level.value} vs ML={ml_level.value}. "
                f"ML detected elevated risk from CDM patterns. Flagged for human review."
            )

        return HybridRiskResult(
            risk_level=final_level,
            confidence=ConfidenceLevel.LOW,
            reason=reason,
            physics_pc=physics_pc,
            physics_risk_level=physics_level,
            physics_reason=physics_reason,
            ml_pc=ml_pc,
            ml_risk_level=ml_level.value if ml_level else None,
            ml_reason=ml_reason,
            needs_human_review=True,
            fusion_method="conservative_disagreement",
        )

    @staticmethod
    def _classify_physics(miss_distance_km: float, pc: float | None) -> tuple[RiskLevel, str]:
        if pc is not None:
            if pc >= 1e-4:
                return RiskLevel.RED, f"Physics Pc={pc:.2e} >= 1e-4"
            if pc >= 1e-6:
                return RiskLevel.YELLOW, f"Physics Pc={pc:.2e} >= 1e-6"
            return RiskLevel.GREEN, f"Physics Pc={pc:.2e} < 1e-6"
        # Fall back to geometric
        if miss_distance_km <= 1.0:
            return RiskLevel.RED, f"Miss distance {miss_distance_km:.3f} km <= 1 km"
        if miss_distance_km <= 5.0:
            return RiskLevel.YELLOW, f"Miss distance {miss_distance_km:.3f} km <= 5 km"
        return RiskLevel.GREEN, f"Miss distance {miss_distance_km:.3f} km > 5 km"

    @staticmethod
    def _classify_ml_pc(pc: float) -> RiskLevel:
        if pc >= 1e-4:
            return RiskLevel.RED
        if pc >= 1e-6:
            return RiskLevel.YELLOW
        return RiskLevel.GREEN

    @staticmethod
    def _str_to_risk_level(level: str) -> RiskLevel:
        mapping = {"RED": RiskLevel.RED, "YELLOW": RiskLevel.YELLOW, "GREEN": RiskLevel.GREEN}
        return mapping.get(level.upper(), RiskLevel.GREEN)

    @staticmethod
    def _risk_order(level: RiskLevel) -> int:
        return {RiskLevel.RED: 0, RiskLevel.YELLOW: 1, RiskLevel.GREEN: 2}.get(level, 9)
