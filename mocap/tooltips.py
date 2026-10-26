"""Shared readable tooltip formatting for static and dynamically updated help."""
from html import escape, unescape

_PREFIX = '<qt><table width="360" cellspacing="0" cellpadding="0"><tr><td>'
_SUFFIX = '</td></tr></table></qt>'


def plain_tooltip(text):
    """Recover our own escaped text when composing dynamic help sections."""
    if text.startswith(_PREFIX) and text.endswith(_SUFFIX):
        return unescape(text[len(_PREFIX):-len(_SUFFIX)].replace('<br/>', '\n'))
    return text


def tooltip_text(text):
    """Let Qt wrap long help at 360 logical pixels; retain explicit line breaks."""
    if not text or text.startswith(_PREFIX):
        return text
    if len(text) <= 64 and '\n' not in text:
        return text
    return _PREFIX + escape(text).replace('\n', '<br/>') + _SUFFIX

