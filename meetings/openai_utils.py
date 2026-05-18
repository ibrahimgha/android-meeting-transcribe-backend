from typing import Any


def chat_completion_options(model: str, *, temperature: float | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {"model": model}
    if temperature is not None and supports_custom_temperature(model):
        options["temperature"] = temperature
    return options


def supports_custom_temperature(model: str) -> bool:
    normalized = model.lower()
    return not (
        normalized == "gpt-5.5"
        or normalized.startswith("gpt-5.")
        or normalized.startswith("gpt-5-")
    )
