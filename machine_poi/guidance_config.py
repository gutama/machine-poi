"""Validated, model-free configuration for reproducible Quran guidance runs."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import EMBEDDING_MODELS, LLM_MODELS
from .corpus import load_verses


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value, minimum, name):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class QuranGuidanceConfig:
    llm: str
    llm_revision: str
    embedding_revision: str
    corpus_path: str
    corpus_sha256: str
    dataset_path: str
    dataset_sha256: str
    embedding: str = "paraphrase-minilm"
    resolutions: list = field(default_factory=lambda: ["verse", "passage"])
    retrieval_k: int = 3
    context_limit: int = 12000
    layers: list = field(default_factory=lambda: [8, 9, 10])
    steering_recipe: str = "contrastive"
    sample_size: int = 32
    training_seed: int = 42
    dose_candidates: list = field(default_factory=lambda: [0, 0.01, 0.02, 0.05])
    theme: str | None = None
    experimental_rotor: bool = False
    rotor_rank: int = 4
    rotor_max_angle_rad: float = 0.1
    rotor_tolerance: float = 1e-6
    displacement_match_tolerance: float = 0.002
    seeds: list = field(default_factory=lambda: [42])
    decoding: dict = field(default_factory=lambda: {
        "max_new_tokens": 256, "do_sample": False, "temperature": 0.7, "top_p": 0.9,
    })
    device: str = "cpu"
    calibration_split: str = "dev"
    training_split: str = "train"
    evaluation_split: str = "test"
    dynamic_steering: bool = False

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(**data)
        result.validate()
        return result

    def validate(self):
        if self.llm not in LLM_MODELS or self.embedding not in EMBEDDING_MODELS:
            raise ValueError("Use registered LLM/embedder identities")
        for revision in (self.llm_revision, self.embedding_revision):
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Reproduction requires full pinned commit revisions")
        for path, digest in ((self.corpus_path, self.corpus_sha256), (self.dataset_path, self.dataset_sha256)):
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or file_hash(path) != digest:
                raise ValueError(f"Stale source hash: {path}")
        verses = load_verses(self.corpus_path)
        # This mode accepts canonical Arabic only. No translation substitution.
        if not all(re.search(r"[\u0600-\u06ff]", v.text) for v in verses):
            raise ValueError("Guidance requires the canonical Arabic corpus")
        if self.dynamic_steering is not False:
            raise ValueError("Frozen guidance does not allow retrieval-dependent steering")
        if (self.training_split, self.calibration_split, self.evaluation_split) != ("train", "dev", "test"):
            raise ValueError("Training, calibration, and evaluation must use train/dev/test")
        if not self.resolutions or len(set(self.resolutions)) != len(self.resolutions) or not set(self.resolutions) <= {"verse", "passage", "surah"}:
            raise ValueError("Invalid retrieval resolutions")
        for value, name in ((self.retrieval_k, "retrieval_k"), (self.context_limit, "context_limit"), (self.sample_size, "sample_size")):
            integer(value, 1, name)
        if self.context_limit > 36000 or self.retrieval_k > 20:
            raise ValueError("Retrieval context/k exceed research bounds")
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Select distinct nonnegative layers")
        for layer in self.layers:
            integer(layer, 0, "layer")
        if self.steering_recipe not in {"centered", "contrastive"}:
            raise ValueError("Unknown steering recipe")
        if not self.dose_candidates or 0 not in self.dose_candidates:
            raise ValueError("Dose sweep must include disabled zero")
        for dose in self.dose_candidates:
            if type(dose) not in (int, float) or not math.isfinite(dose) or not 0 <= dose <= 1:
                raise ValueError("Invalid experimental dose candidate")
        if type(self.experimental_rotor) is not bool:
            raise ValueError("experimental_rotor must be boolean")
        integer(self.training_seed, 0, "training_seed")
        integer(self.rotor_rank, 2, "rotor_rank")
        for value, upper, name in ((self.rotor_max_angle_rad, math.pi, "rotor angle"),
                                   (self.rotor_tolerance, 1e-3, "rotor tolerance"),
                                   (self.displacement_match_tolerance, 0.1, "match tolerance")):
            if not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError(f"Invalid {name}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Select distinct seeds")
        for seed in self.seeds:
            integer(seed, 0, "seed")
        if set(self.decoding) != {"max_new_tokens", "do_sample", "temperature", "top_p"}:
            raise ValueError("Specify exactly the supported decoding settings")
        integer(self.decoding["max_new_tokens"], 1, "max_new_tokens")
        if type(self.decoding["do_sample"]) is not bool:
            raise ValueError("do_sample must be boolean")
        for key, upper in (("temperature", 10), ("top_p", 1)):
            if not math.isfinite(self.decoding[key]) or not 0 < self.decoding[key] <= upper:
                raise ValueError(f"Invalid {key}")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Invalid device")
        from .behavior_data import load_behavior_data
        examples = load_behavior_data(self.dataset_path, self.corpus_path, self.theme)
        if self.experimental_rotor and self.rotor_rank > 2 * sum(r["split"] == "train" for r in examples):
            raise ValueError("Rotor rank exceeds available training pairs")
        return self

    def resolved(self):
        self.validate()
        return {**asdict(self), "llm_identity": LLM_MODELS[self.llm]["hf_path"],
                "embedder_identity": EMBEDDING_MODELS[self.embedding]["hf_path"],
                "dose_status": "experimental candidates, not validated safety thresholds"}
