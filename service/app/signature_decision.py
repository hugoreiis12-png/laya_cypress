"""Deterministic signature engine: turns evidence into one of the four result states, apart from any model."""
from dataclasses import dataclass

from . import signature
from .config import Settings
from .signature_model import ACTION_REVIEW, ACTION_SIGNED, SignatureDecision
from .vision.types import InkFeatures
from .vision_judge import SignatureOpinion

SIGNED = "SIGNED"
UNSIGNED = "UNSIGNED"
UNCERTAIN = "UNCERTAIN"
INVALID_IMAGE = "INVALID_IMAGE"
PERCENT = 100
TWO_VOTES_CONFIDENCE = 95.0  # above the n8n If4 cutoff of 90: only two independent votes may release a canhoto
JUDGE_RELEASE_CONFIDENCE = 95.0  # above the n8n If4 cutoff of 90, whatever confidence above the minimum the judge gave


@dataclass(frozen=True)
class SignatureVerdict:
    state: str
    confidence: float

    @property
    def signed(self) -> bool:
        return self.state == SIGNED


UNDECIDED = SignatureVerdict(UNCERTAIN, 0.0)


def from_ink(features: InkFeatures, settings: Settings) -> SignatureVerdict:
    """A receiver field that was not found proves nothing, so it never counts as a confirmed absence."""
    if not features.region_found:
        return UNDECIDED
    signed, confidence = signature.decide(features, settings)
    return SignatureVerdict(SIGNED if signed else UNSIGNED, confidence)


def from_model(decision: SignatureDecision) -> SignatureVerdict:
    if decision.action == ACTION_REVIEW:
        return UNDECIDED
    if decision.action == ACTION_SIGNED:
        return SignatureVerdict(SIGNED, PERCENT * decision.probability_signed)
    return SignatureVerdict(UNSIGNED, PERCENT * (1 - decision.probability_signed))


def from_votes(detector_signed: bool | None, opinion: SignatureOpinion | None, min_confidence: float) -> SignatureVerdict:
    """Anything short of two confident votes for a signature stays undecided and goes to the reserve."""
    judge_signed = opinion is not None and opinion.signed is True and opinion.confidence >= min_confidence
    return SignatureVerdict(SIGNED, TWO_VOTES_CONFIDENCE) if detector_signed is True and judge_signed else UNDECIDED


def from_judge_alone(opinion: SignatureOpinion | None, min_confidence: float) -> SignatureVerdict:
    """A confident signed opinion is released; a "no" is not proof either, so it also goes to the reserve."""
    if opinion is None or opinion.signed is not True or opinion.confidence < min_confidence:
        return UNDECIDED
    return SignatureVerdict(SIGNED, JUDGE_RELEASE_CONFIDENCE)


def from_judge(opinion: SignatureOpinion, max_confidence: float) -> SignatureVerdict:
    """The cap keeps an unmeasured judge below the n8n release cutoff; abstaining stays undecided."""
    if opinion.signed is None:
        return UNDECIDED
    return SignatureVerdict(SIGNED if opinion.signed else UNSIGNED, min(opinion.confidence, max_confidence))
