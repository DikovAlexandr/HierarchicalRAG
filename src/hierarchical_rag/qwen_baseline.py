"""Pure chat-prompt and answer parsing for the frozen Qwen CoT baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from hierarchical_rag.hotpotqa import (
    HotpotExample,
    Paragraph,
    gold_paragraphs,
    serialize_context,
)


SYSTEM_INSTRUCTION = (
    "Use only the provided evidence. Reason briefly and explicitly. "
    'End with exactly one final line in the form "Answer: <shortest answer>".'
)


@dataclass(frozen=True, slots=True)
class ChatPromptBuild:
    prompt: str
    input_tokens: int
    included_paragraphs: tuple[Paragraph, ...]
    included_document_count: int
    included_sentence_count: int
    dropped_document_count: int
    dropped_sentence_count: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class BaselineAnswerExtraction:
    answer: str | None
    status: str
    method: str


def build_cot_chat_prompt(
    demonstrations: Sequence[HotpotExample],
    target: HotpotExample,
    target_paragraphs: Sequence[Paragraph],
    *,
    render_chat: Callable[[Sequence[Mapping[str, str]]], str],
    token_count: Callable[[str], int],
    max_input_tokens: int,
) -> ChatPromptBuild:
    """Render the fixed few-shot CoT chat and truncate target evidence only."""

    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive")
    if len(demonstrations) != 2:
        raise ValueError("D010 requires exactly two demonstrations")
    for demonstration in demonstrations:
        if demonstration.answer is None:
            raise ValueError(
                f"{demonstration.identifier}: demonstration answer is unavailable"
            )
        if not gold_paragraphs(demonstration):
            raise ValueError(
                f"{demonstration.identifier}: demonstration has no gold paragraphs"
            )
    if not target_paragraphs:
        raise ValueError(f"{target.identifier}: target evidence is empty")

    full_prompt = _render(
        demonstrations,
        target,
        target_paragraphs,
        render_chat=render_chat,
    )
    full_tokens = token_count(full_prompt)
    total_sentences = sum(len(paragraph.sentences) for paragraph in target_paragraphs)
    if full_tokens <= max_input_tokens:
        return ChatPromptBuild(
            prompt=full_prompt,
            input_tokens=full_tokens,
            included_paragraphs=tuple(target_paragraphs),
            included_document_count=len(target_paragraphs),
            included_sentence_count=total_sentences,
            dropped_document_count=0,
            dropped_sentence_count=0,
            truncated=False,
        )

    included: list[Paragraph] = []
    included_sentence_count = 0
    exhausted = False
    for paragraph in target_paragraphs:
        accepted_sentences: list[str] = []
        for sentence in paragraph.sentences:
            candidate_paragraph = Paragraph(
                title=paragraph.title,
                sentences=tuple((*accepted_sentences, sentence)),
            )
            candidate_prompt = _render(
                demonstrations,
                target,
                (*included, candidate_paragraph),
                render_chat=render_chat,
            )
            if token_count(candidate_prompt) > max_input_tokens:
                exhausted = True
                break
            accepted_sentences.append(sentence)
            included_sentence_count += 1
        if accepted_sentences:
            included.append(
                Paragraph(
                    title=paragraph.title,
                    sentences=tuple(accepted_sentences),
                )
            )
        if exhausted:
            break

    if not included:
        raise ValueError(
            f"{target.identifier}: no target evidence sentence fits the input budget"
        )
    prompt = _render(
        demonstrations,
        target,
        included,
        render_chat=render_chat,
    )
    input_tokens = token_count(prompt)
    if input_tokens > max_input_tokens:
        raise RuntimeError("sentence-boundary truncation exceeded the input budget")
    return ChatPromptBuild(
        prompt=prompt,
        input_tokens=input_tokens,
        included_paragraphs=tuple(included),
        included_document_count=len(included),
        included_sentence_count=included_sentence_count,
        dropped_document_count=len(target_paragraphs) - len(included),
        dropped_sentence_count=total_sentences - included_sentence_count,
        truncated=True,
    )


def extract_final_answer(generated_text: str) -> BaselineAnswerExtraction:
    """Extract only the predeclared final answer line without semantic repair."""

    text = generated_text.strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(marker, 1)[0].strip()
    if not text:
        return BaselineAnswerExtraction(None, "empty", "none")

    matches = re.findall(
        r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$",
        text,
    )
    if not matches:
        return BaselineAnswerExtraction(None, "missing_final_answer", "none")
    answer = _unwrap_box(matches[-1])
    if not answer:
        return BaselineAnswerExtraction(None, "empty_final_answer", "answer_line")
    return BaselineAnswerExtraction(answer, "ok", "last_answer_line")


def has_explicit_reasoning(generated_text: str) -> bool:
    """Return whether non-empty text precedes the declared final answer line."""

    text = generated_text
    for marker in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(marker, 1)[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    answer_line = re.compile(r"(?i)^(?:final\s+)?answer\s*:")
    return any(not answer_line.match(line) for line in lines)


def _render(
    demonstrations: Sequence[HotpotExample],
    target: HotpotExample,
    target_paragraphs: Sequence[Paragraph],
    *,
    render_chat: Callable[[Sequence[Mapping[str, str]]], str],
) -> str:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION}
    ]
    for demonstration in demonstrations:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": serialize_context(
                        demonstration.question,
                        gold_paragraphs(demonstration),
                    ).rstrip(),
                },
                {
                    "role": "assistant",
                    "content": f"Answer: {demonstration.answer}",
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": serialize_context(target.question, target_paragraphs).rstrip(),
        }
    )
    return render_chat(messages)


def _unwrap_box(value: str) -> str:
    answer = value.strip().strip("$").strip()
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        answer = answer[len("\\boxed{") : -1].strip()
    if answer.startswith("\\text{") and answer.endswith("}"):
        answer = answer[len("\\text{") : -1].strip()
    return answer
