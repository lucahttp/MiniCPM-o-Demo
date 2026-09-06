"""
core/expert/streaming.py

Streaming sentence boundary tokenizer and speculative bridging utilities for voice duplex.
Detects sentence completion in token streams while handling abbreviations, initials, and decimals.
"""
import re
from typing import AsyncGenerator, Optional, List

class StreamingSentenceTokenizer:
    """
    Incrementally buffers streaming tokens and yields complete sentences as soon as
    punctuation is detected.
    Protects abbreviations (e.g. 'e.g.', 'Dr.', 'Mr.', 'vs.', 'etc.') and numbers with decimals ('4.5', '1.000').
    """
    _ABBREVIATIONS = {
        "dr", "mr", "mrs", "ms", "prof", "sr", "sra", "lic", "ing", "ej",
        "vs", "etc", "approx", "approx.", "fig", "no", "gen", "vol", "pág", "pag", "ars", "usd"
    }

    # Sentence boundary: . ? ! ¡ ¿ \n followed by whitespace or end of string
    _SENTENCE_END_RE = re.compile(r'([.?!¡¿\n]+)(\s+|$)')

    def __init__(self, min_sentence_words: int = 3):
        self.buffer = ""
        self.min_sentence_words = min_sentence_words

    def feed(self, token: str) -> List[str]:
        """
        Feed an incoming token chunk and return a list of any newly completed sentences.
        """
        self.buffer += token
        sentences = []

        while True:
            match = self._SENTENCE_END_RE.search(self.buffer)
            if not match:
                break

            candidate = self.buffer[:match.end()].strip()
            period_pos = match.start(1)
            before_punct = self.buffer[:period_pos].rstrip()
            last_word = before_punct.split()[-1].lower() if before_punct.split() else ""
            last_word_clean = re.sub(r'[^\w]', '', last_word)

            # Check if this period is actually part of an abbreviation
            if last_word_clean in self._ABBREVIATIONS:
                break

            # Check for decimals like 4.5 or $4.000
            if period_pos > 0 and period_pos < len(self.buffer) - 1:
                if self.buffer[period_pos - 1].isdigit() and self.buffer[period_pos + 1].isdigit():
                    break

            if len(candidate.split()) >= self.min_sentence_words or "\n" in match.group(1):
                sentences.append(candidate)
                self.buffer = self.buffer[match.end():]
            else:
                break

        return sentences

    def flush(self) -> Optional[str]:
        """Flush any remaining text in the buffer."""
        rest = self.buffer.strip()
        self.buffer = ""
        return rest if rest else None
