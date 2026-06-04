"""Model configuration for OpenAI-compatible benchmark runs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from eval.paths import configure_ca_bundle


LOCAL_API_BASE_PREFIXES = (
    "http://127.0.0.1:",
    "http://localhost:",
    "http://0.0.0.0:",
    "http://172.17.0.1.nip.io:",
)


@dataclass
class ModelConfig:
    name: str
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    litellm_model: str | None = None
    temperature: float = 0
    max_tokens: int = 4096
    extra_body: dict[str, Any] = field(default_factory=dict)
    require_api_key: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        name = data.get("name") or data.get("model")
        if not name:
            raise ValueError("model config must include `name` or `model`")
        extra_body = data.get("extra_body") or {}
        if isinstance(extra_body, str):
            extra_body = json.loads(extra_body)
        return cls(
            name=name,
            api_base=data.get("api_base"),
            api_key_env=data.get("api_key_env", "OPENAI_API_KEY"),
            litellm_model=data.get("litellm_model"),
            temperature=data.get("temperature", 0),
            max_tokens=data.get("max_tokens", 4096),
            extra_body=extra_body,
            require_api_key=data.get("require_api_key"),
        )

    def with_overrides(
        self,
        *,
        name: str | None = None,
        api_base: str | None = None,
        api_key_env: str | None = None,
        litellm_model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body_json: str | None = None,
        require_api_key: bool | None = None,
    ) -> "ModelConfig":
        extra_body = self.extra_body
        if extra_body_json:
            extra_body = json.loads(extra_body_json)
        return ModelConfig(
            name=name or self.name,
            api_base=api_base if api_base is not None else self.api_base,
            api_key_env=api_key_env or self.api_key_env,
            litellm_model=litellm_model if litellm_model is not None else self.litellm_model,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            extra_body=extra_body,
            require_api_key=require_api_key if require_api_key is not None else self.require_api_key,
        )

    @property
    def openai_model(self) -> str:
        if self.name.startswith("openai/"):
            return self.name.removeprefix("openai/")
        return self.name

    @property
    def pier_model(self) -> str:
        return self.litellm_model or (self.name if self.name.startswith("openai/") else f"openai/{self.name}")

    @property
    def litellm_name(self) -> str:
        return self.litellm_model or (self.name if self.name.startswith("openai/") else f"openai/{self.name}")

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", self.openai_model).strip("-").lower()

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if key:
            return key
        if self.require_api_key is False or self.is_local_api_base:
            return "dummy"
        raise RuntimeError(f"{self.api_key_env} must be set in the environment")

    @property
    def is_local_api_base(self) -> bool:
        return bool(self.api_base and self.api_base.startswith(LOCAL_API_BASE_PREFIXES))

    def generation_env(self) -> dict[str, str]:
        env = os.environ.copy()
        configure_ca_bundle(env)
        env["OPENAI_API_KEY"] = self.api_key()
        if self.api_base:
            env["OPENAI_BASE_URL"] = self.api_base
            env["OPENAI_API_BASE"] = self.api_base
        env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
        return env

    def to_cli_args(self) -> list[str]:
        args = [
            "--model",
            self.name,
            "--api-key-env",
            self.api_key_env,
            "--api-base",
            self.api_base or "",
            "--litellm-model",
            self.litellm_model or "",
        ]
        args.extend(["--temperature", str(self.temperature)])
        args.extend(["--max-tokens", str(self.max_tokens)])
        args.extend(["--extra-body-json", json.dumps(self.extra_body, separators=(",", ":"))])
        if self.require_api_key is False:
            args.append("--no-require-api-key")
        elif self.require_api_key is True:
            args.append("--require-api-key")
        return args


def add_model_args(parser) -> None:
    parser.add_argument("--model", help="Model name to send to the OpenAI-compatible API.")
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key-env", help="Environment variable containing the API key.")
    parser.add_argument("--litellm-model", help="LiteLLM/Pier model name, e.g. openai/Qwen/Qwen3-8B.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, help="Maximum output tokens.")
    parser.add_argument("--extra-body-json", help="JSON object passed as provider-specific extra_body.")
    parser.add_argument(
        "--require-api-key",
        dest="require_api_key",
        action="store_true",
        default=None,
        help="Require an API key even for local-looking endpoints.",
    )
    parser.add_argument(
        "--no-require-api-key",
        dest="require_api_key",
        action="store_false",
        help="Allow no-auth OpenAI-compatible endpoints.",
    )


def model_from_defaults(defaults: dict[str, Any], args=None) -> ModelConfig:
    model_data = defaults.get("model_config")
    if model_data is None:
        model_data = {
            "name": defaults.get("model", "gpt-4.1-mini"),
            "api_base": defaults.get("api_base"),
            "api_key_env": defaults.get("api_key_env", "OPENAI_API_KEY"),
            "litellm_model": defaults.get("litellm_model"),
            "temperature": defaults.get("temperature", 0),
            "max_tokens": defaults.get("max_tokens", 4096),
            "extra_body": defaults.get("extra_body", {}),
        }
    config = ModelConfig.from_dict(model_data)
    if args is None:
        return config
    return config.with_overrides(
        name=getattr(args, "model", None),
        api_base=getattr(args, "api_base", None),
        api_key_env=getattr(args, "api_key_env", None),
        litellm_model=getattr(args, "litellm_model", None),
        temperature=getattr(args, "temperature", None),
        max_tokens=getattr(args, "max_tokens", None),
        extra_body_json=getattr(args, "extra_body_json", None),
        require_api_key=getattr(args, "require_api_key", None),
    )
