"""Pure prompt and answer-interface logic for the frozen HRM-Text reader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from hierarchical_rag.hotpotqa import (
    HotpotExample,
    Paragraph,
    gold_paragraphs,
    serialize_context,
)


DIRECT_CONDITION = "<|object_ref_start|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PROMPT_INSTRUCTION = (
    "Answer each question using only its evidence. "
    'Return only the shortest answer after "Answer:".'
)


@dataclass(frozen=True, slots=True)
class PromptBuild:
    prompt: str
    input_tokens: int
    included_paragraphs: tuple[Paragraph, ...]
    included_document_count: int
    included_sentence_count: int
    dropped_document_count: int
    dropped_sentence_count: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class AnswerExtraction:
    answer: str | None
    status: str
    method: str


def build_direct_prompt(
    demonstrations: Sequence[HotpotExample],
    target: HotpotExample,
    target_paragraphs: Sequence[Paragraph],
    *,
    token_count: Callable[[str], int],
    max_input_tokens: int,
) -> PromptBuild:
    """Build the fixed direct/few-shot prompt and truncate target evidence only."""

    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive")
    if not demonstrations:
        raise ValueError("at least one demonstration is required")
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

    full_prompt = _render_prompt(demonstrations, target, target_paragraphs)
    full_tokens = token_count(full_prompt)
    total_sentences = sum(len(paragraph.sentences) for paragraph in target_paragraphs)
    if full_tokens <= max_input_tokens:
        return PromptBuild(
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
            candidate_prompt = _render_prompt(
                demonstrations,
                target,
                (*included, candidate_paragraph),
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
    prompt = _render_prompt(demonstrations, target, included)
    input_tokens = token_count(prompt)
    if input_tokens > max_input_tokens:
        raise RuntimeError("sentence-boundary truncation exceeded the input budget")
    return PromptBuild(
        prompt=prompt,
        input_tokens=input_tokens,
        included_paragraphs=tuple(included),
        included_document_count=len(included),
        included_sentence_count=included_sentence_count,
        dropped_document_count=len(target_paragraphs) - len(included),
        dropped_sentence_count=total_sentences - included_sentence_count,
        truncated=True,
    )


def extract_short_answer(generated_text: str) -> AnswerExtraction:
    """Apply only predeclared structural extraction, never semantic rewriting."""

    text = generated_text.strip()
    for marker in ("<|box_end|>", IM_END, "<|endoftext|>"):
        text = text.split(marker, 1)[0].strip()
    if not text:
        return AnswerExtraction(answer=None, status="empty", method="none")

    boxed = _boxed_content(text)
    if boxed is not None:
        answer = _unwrap_latex_text(boxed)
        if answer:
            return AnswerExtraction(
                answer=answer,
                status="ok",
                method="latex_boxed",
            )

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    for prefix in ("Answer:", "Final answer:"):
        if first_line.casefold().startswith(prefix.casefold()):
            first_line = first_line[len(prefix) :].strip()
            break
    first_line = first_line.strip().strip("$").strip()
    if not first_line:
        return AnswerExtraction(answer=None, status="empty", method="none")
    return AnswerExtraction(answer=first_line, status="ok", method="first_line")


def _render_prompt(
    demonstrations: Sequence[HotpotExample],
    target: HotpotExample,
    target_paragraphs: Sequence[Paragraph],
) -> str:
    blocks = [PROMPT_INSTRUCTION]
    for demonstration in demonstrations:
        context = serialize_context(
            demonstration.question,
            gold_paragraphs(demonstration),
        ).rstrip()
        blocks.append(f"{context}\nAnswer: {demonstration.answer}")
    target_context = serialize_context(target.question, target_paragraphs).rstrip()
    blocks.append(f"{target_context}\nAnswer:")
    body = "\n\n".join(blocks)
    return f"{IM_START}{DIRECT_CONDITION}{body}{IM_END}"


def _boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.find(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index].strip()
    return None


def _unwrap_latex_text(value: str) -> str:
    answer = value.strip().strip("$").strip()
    for command in ("\\text{", "\\mathrm{", "\\operatorname{"):
        if answer.startswith(command) and answer.endswith("}"):
            answer = answer[len(command) : -1].strip()
            break
    return answer
