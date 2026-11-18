from __future__ import annotations
import io
import keyword
import tokenize

_TYPESCRIPT_KEYWORDS = {
    "abstract",
    "any",
    "as",
    "asserts",
    "async",
    "await",
    "bigint",
    "boolean",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "constructor",
    "continue",
    "debugger",
    "declare",
    "default",
    "delete",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "from",
    "function",
    "get",
    "if",
    "implements",
    "import",
    "in",
    "infer",
    "instanceof",
    "interface",
    "is",
    "keyof",
    "let",
    "module",
    "namespace",
    "never",
    "new",
    "null",
    "number",
    "object",
    "of",
    "package",
    "private",
    "protected",
    "public",
    "readonly",
    "require",
    "return",
    "set",
    "static",
    "string",
    "super",
    "switch",
    "symbol",
    "this",
    "throw",
    "true",
    "try",
    "type",
    "typeof",
    "undefined",
    "unique",
    "unknown",
    "var",
    "void",
    "while",
    "with",
    "yield",
}


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    pos = 0
    for part in text.splitlines(True):
        pos += len(part)
        offsets.append(pos)
    return offsets


def _absolute_position(text: str, start: tuple[int, int]) -> int | None:
    (row, col) = start
    offsets = _line_offsets(text)
    if row <= 0 or row > len(offsets):
        return None
    return offsets[row - 1] + col


def _first_nonkeyword_python_name(text: str) -> tuple[int, str, tuple[int, int]] | None:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type != tokenize.NAME or keyword.iskeyword(token.string):
                continue
            position = _absolute_position(text, token.start)
            if position is not None and 0 <= position < len(text):
                return (position, token.string, token.start)
    except (IndentationError, SyntaxError, SystemError, tokenize.TokenError):
        return None
    return None


def _is_typescript_identifier_start(char: str) -> bool:
    return char == "$" or char == "_" or char.isalpha()


def _is_typescript_identifier_part(char: str) -> bool:
    return char == "$" or char == "_" or char.isalnum()


def _first_nonkeyword_typescript_name(
    text: str,
) -> tuple[int, str, tuple[int, int]] | None:
    index = 0
    length = len(text)
    line = 1
    col = 0
    while index < length:
        char = text[index]
        next_char = text[index + 1] if index + 1 < length else ""
        if char == "\n":
            index += 1
            line += 1
            col = 0
            continue
        if char == "/" and next_char == "/":
            newline = text.find("\n", index + 2)
            if newline < 0:
                return None
            col += newline + 1 - index
            index = newline + 1
            line += 1
            col = 0
            continue
        if char == "/" and next_char == "*":
            end = text.find("*/", index + 2)
            block = text[index:] if end < 0 else text[index : end + 2]
            for block_char in block:
                if block_char == "\n":
                    line += 1
                    col = 0
                else:
                    col += 1
            index = length if end < 0 else end + 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            col += 1
            while index < length:
                current = text[index]
                if current == "\\":
                    step = min(2, length - index)
                    index += step
                    col += step
                    continue
                index += 1
                if current == "\n":
                    line += 1
                    col = 0
                else:
                    col += 1
                if current == quote:
                    break
            continue
        if not _is_typescript_identifier_start(char):
            index += 1
            col += 1
            continue
        start = index
        start_line = line
        start_col = col
        index += 1
        col += 1
        while index < length and _is_typescript_identifier_part(text[index]):
            index += 1
            col += 1
        token_text = text[start:index]
        if token_text not in _TYPESCRIPT_KEYWORDS:
            return (start, token_text, (start_line, start_col))
    return None


def _first_nonkeyword_name(
    text: str, *, language: str
) -> tuple[int, str, tuple[int, int]] | None:
    if language == "python":
        return _first_nonkeyword_python_name(text)
    if language == "typescript":
        return _first_nonkeyword_typescript_name(text)
    raise ValueError(f"Unsupported language: {language}")
