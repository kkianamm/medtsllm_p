#!/usr/bin/env python3
"""Generate feature-only PULSE descriptions for rendered PTB-XL images.

Run this script inside the separate PULSE/LLaVA environment. The PULSE model is
loaded once and reused for all images. Results are appended to JSONL so an
interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image


ALLOWED_FIELDS = (
    "rhythm",
    "rate",
    "axis",
    "p_wave",
    "pr_interval",
    "qrs",
    "st_segment",
    "t_wave",
    "ectopy",
    "image_quality",
    "uncertainty",
)

FEATURE_ONLY_PROMPT = """
Analyze this 12-lead ECG image and return only one valid JSON object.
Describe visual ECG findings, not a final diagnosis and not a dataset class.
Use "unknown" whenever a feature cannot be determined reliably.
Do not output NORM, MI, STTC, CD, HYP, a superclass label, treatment advice,
or explanatory text outside the JSON object.

Use exactly these keys:
{
  "rhythm": "...",
  "rate": "...",
  "axis": "...",
  "p_wave": "...",
  "pr_interval": "...",
  "qrs": "...",
  "st_segment": "...",
  "t_wave": "...",
  "ectopy": "...",
  "image_quality": "...",
  "uncertainty": "low|moderate|high"
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulse-root", type=Path, required=True, help="Path to PULSE/LLaVA")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", default="PULSE-ECG/PULSE-7B")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def completed_ids(path: Path) -> set[int]:
    """Return IDs whose latest successful generation is already available."""
    if not path.exists():
        return set()
    latest_status: dict[int, str] = {}
    for record in read_jsonl(path):
        if "ecg_id" in record:
            latest_status[int(record["ecg_id"])] = str(record.get("status", "ok"))
    return {ecg_id for ecg_id, status in latest_status.items() if status == "ok"}


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(cleaned[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("PULSE response did not contain a valid JSON object")


def normalize_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "unknown"


def sanitize_structured(raw: dict[str, Any]) -> dict[str, str]:
    return {field: normalize_value(raw.get(field, "unknown")) for field in ALLOWED_FIELDS}


def format_description(structured: dict[str, str]) -> str:
    pieces = [f"{field.replace('_', ' ')}={structured[field]}" for field in ALLOWED_FIELDS]
    return "PULSE visual ECG findings (feature-only): " + "; ".join(pieces) + "."


def move_images_to_device(images: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(images, list):
        return [image.to(device=device, dtype=dtype) for image in images]
    return images.to(device=device, dtype=dtype)


def build_prompt(model: Any, conv_templates: Any, constants: Any, conv_mode: str) -> str:
    image_token = constants.DEFAULT_IMAGE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        image_token = constants.DEFAULT_IM_START_TOKEN + image_token + constants.DEFAULT_IM_END_TOKEN
    query = image_token + "\n" + FEATURE_ONLY_PROMPT
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], query)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def main() -> None:
    args = parse_args()
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    if args.load_4bit and args.load_8bit:
        raise ValueError("Choose at most one of --load-4bit and --load-8bit")

    pulse_root = args.pulse_root.expanduser().resolve()
    if not (pulse_root / "llava").exists():
        raise FileNotFoundError(f"Expected a PULSE/LLaVA tree at {pulse_root}")
    sys.path.insert(0, str(pulse_root))

    from llava import constants
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
    )
    model.eval()
    prompt = build_prompt(model, conv_templates, constants, args.conv_mode)
    prompt_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        constants.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)

    manifest = read_jsonl(args.manifest.expanduser().resolve())
    manifest = [record for index, record in enumerate(manifest) if index % args.world_size == args.rank]
    if args.limit is not None:
        manifest = manifest[: args.limit]

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(output_path)
    remaining = [record for record in manifest if int(record["ecg_id"]) not in done]
    print(f"rank={args.rank}/{args.world_size}; completed={len(done)}; remaining={len(remaining)}")

    model_device = next(model.parameters()).device
    image_dtype = torch.float16 if model_device.type == "cuda" else torch.float32
    input_ids = prompt_ids.to(model_device)

    with output_path.open("a", encoding="utf-8", buffering=1) as output_handle:
        for index, record in enumerate(remaining, start=1):
            ecg_id = int(record["ecg_id"])
            image_path = Path(record["image_path"]).expanduser()
            result: dict[str, Any] = {
                "ecg_id": ecg_id,
                "split": record.get("split"),
                "image_path": str(image_path),
                "model_path": args.model_path,
                "prompt_type": "feature_only_v1",
            }
            try:
                image = Image.open(image_path).convert("RGB")
                image_sizes = [image.size]
                images = process_images([image], image_processor, model.config)
                images = move_images_to_device(images, model_device, image_dtype)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images,
                        image_sizes=image_sizes,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                # Hugging Face causal generation normally returns prompt + continuation.
                # Decode only the continuation so the JSON schema in our prompt cannot
                # be mistaken for the model's answer.
                if output_ids.shape[1] > input_ids.shape[1]:
                    generated_ids = output_ids[:, input_ids.shape[1] :]
                else:
                    generated_ids = output_ids
                raw_output = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
                parsed = extract_json_object(raw_output)
                structured = sanitize_structured(parsed)
                result.update(
                    {
                        "status": "ok",
                        "structured": structured,
                        "description": format_description(structured),
                        "raw_output": raw_output,
                    }
                )
            except Exception as exc:  # keep long jobs resumable and auditable
                result.update(
                    {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=5),
                    }
                )
                if args.fail_fast:
                    output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    raise

            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(remaining):
                print(f"Processed {index}/{len(remaining)} on rank {args.rank}")

    print(f"Descriptions appended to {output_path}")


if __name__ == "__main__":
    main()
