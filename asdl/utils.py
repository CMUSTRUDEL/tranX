# coding=utf-8
import re

__all__ = [
    "remove_comment",
]


def remove_comment(text: str) -> str:
    text = re.sub(re.compile("#.*"), "", text)
    text = '\n'.join(filter(lambda x: x, text.split('\n')))

    return text
