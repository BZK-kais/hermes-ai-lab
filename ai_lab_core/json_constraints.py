"""Validate and compile a bounded set of canonical JSON responses."""
from dataclasses import dataclass, field
import json
import math

MAX_CHOICE_BYTES = 1024
MAX_TOTAL_BYTES = 32_768
MAX_DEPTH = 8
INVALID_SPEC = "invalid_output_constraint"
INVALID_STREAM = "invalid_generation_stream"


class ConstraintStreamError(ValueError):
    pass


def _check(condition):
    if not condition:
        raise ValueError(INVALID_SPEC)


def _check_value(value, depth, remaining):
    # Charge a lower bound before serializing, including on in-memory inputs.
    kind = type(value)
    if kind is str:
        remaining -= len(value) + 2
    else:
        remaining -= 1
    _check(remaining >= 0)
    if kind is dict or kind is list:
        _check(depth <= MAX_DEPTH and len(value) <= remaining)
        if kind is dict:
            for key, item in value.items():
                _check(type(key) is str)
                remaining = _check_value(key, depth + 1, remaining)
                remaining = _check_value(item, depth + 1, remaining)
        else:
            for item in value:
                remaining = _check_value(item, depth + 1, remaining)
    elif kind is float:
        _check(math.isfinite(value))
    elif kind is int:
        _check(value.bit_length() <= MAX_CHOICE_BYTES * 4)
    else:
        _check(kind in (str, bool, type(None)))
    return remaining


def canonical_choices(spec):
    """Return canonical UTF-8-compatible JSON objects in declaration order."""
    _check(type(spec) is dict and len(spec) == 2 and all(type(key) is str for key in spec)
           and spec.keys() == {"kind", "values"})
    _check(type(spec["kind"]) is str and spec["kind"] == "json_choices")
    values = spec["values"]
    _check(type(values) is list and 2 <= len(values) <= 32)
    choices = []
    total = 0
    for value in values:
        _check(type(value) is dict)
        _check_value(value, 1, MAX_CHOICE_BYTES)
        try:
            choice = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            size = len(choice.encode("utf-8"))
        except (UnicodeError, ValueError):
            raise ValueError(INVALID_SPEC) from None
        total += size
        _check(size <= MAX_CHOICE_BYTES and total <= MAX_TOTAL_BYTES and choice not in choices)
        choices.append(choice)
    return tuple(choices)


@dataclass
class _Node:
    children: dict = field(default_factory=dict)
    text: str = None


@dataclass
class _Decoder:
    mx: object
    root: _Node
    eos: tuple

    def generation(self, prompt_length):
        """Build a fresh processor and stream validator for one prompt."""
        processed = set()

        def processor(tokens, logits):
            if len(tokens) < prompt_length:
                raise ConstraintStreamError(INVALID_STREAM)
            prefix = tuple(tokens[prompt_length:].tolist())
            node = self.root
            terminal = None
            for token in prefix:
                if node.text is not None and token in self.eos and terminal in (None, token):
                    # MLX computes one lookahead step before returning its EOS.
                    terminal = token
                    continue
                node = node.children.get(token)
                if node is None:
                    raise ConstraintStreamError(INVALID_STREAM)
            processed.add(prefix)
            if terminal is not None:
                allowed = (terminal,)
            else:
                allowed = self.eos if node.text is not None else tuple(node.children)
            if max(allowed) >= logits.shape[-1]:
                raise ConstraintStreamError(INVALID_STREAM)
            indexes = self.mx.array(allowed)
            masked = self.mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
            masked[:, indexes] = logits[:, indexes]
            return masked

        def verify(tokens, output, finish_reason):
            if finish_reason != "stop" or not tokens or tokens[-1] not in self.eos:
                return False
            node = self.root
            for index, token in enumerate(tokens):
                if type(token) is not int or tuple(tokens[:index]) not in processed:
                    return False
                if index < len(tokens) - 1:
                    node = node.children.get(token)
                    if node is None:
                        return False
            return node.text == output

        return processor, verify


def compile_constraint(mx, tokenizer, spec, max_tokens):
    choices = canonical_choices(spec)
    root = _Node()
    try:
        eos = set(tokenizer.eos_token_ids)
        _check(eos and all(type(token) is int and token >= 0 for token in eos))
        special = set(tokenizer.all_special_ids) | eos
        detokenizer = tokenizer.detokenizer
        for choice in choices:
            ids = tokenizer.encode(choice, add_special_tokens=False)
            _check(type(ids) is list and 0 < len(ids) < max_tokens)
            _check(all(type(token) is int and token >= 0 and token not in special for token in ids))
            detokenizer.reset()
            node = root
            for token in ids:
                detokenizer.add_token(token)
                _check(node.text is None)
                node = node.children.setdefault(token, _Node())
            detokenizer.finalize()
            _check(detokenizer.text == choice and node.text is None and not node.children)
            node.text = choice
    except Exception:
        raise ValueError(INVALID_SPEC) from None
    return _Decoder(mx, root, tuple(sorted(eos)))
