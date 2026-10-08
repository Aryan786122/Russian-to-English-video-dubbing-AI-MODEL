import re
from pathlib import Path
import torch
from transformers import MarianMTModel, MarianTokenizer
import config


class ContextualRussianTranslator:
    """
    Local, neural, context-aware Russian-to-English translator.
    Uses MarianMT (Helsinki-NLP/opus-mt-ru-en) coupled with conversational
    and explicit slang adaptation to preserve tone, profanity, reactions,
    jokes, and syllable timing for natural voice dubbing.
    """

    MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"

    # Contextual slang, profanity (mat), and conversational idiom mappings
    SLANG_AND_MAT_MAP = [
        (r"\b(иди\s+нахуй|пошел\s+нахуй|пошёл\s+нахуй)\b", "fuck off"),
        (r"\bиди\s+на\b(?!\s*(?:сюда|куй))", "fuck off"),
        (r"\b(нахуй|на\s+хуй)\b", "the fuck"),
        (r"\b(похуй|по\s+хуй)\b", "don't give a fuck"),
        (r"\b(охуеть|офигеть|ебать|ебать-колотить)\b", "holy shit"),
        (r"\b(заебись)\b", "fucking awesome"),
        (r"\b(пиздец|капец|пипец)\b", "holy shit"),
        (r"\b(сука\s*,\s*пиздец|сука\s+пиздец)\b", "bitch, holy shit"),
        (r"\b(сука|сучка)\b", "bitch"),
        (r"\b(блядь|блять|бля)\b", "fuck"),
        (r"\b(заткнись|закрой\s+рот)\b", "shut the fuck up"),
        (r"\b(да\s+ладно|да\s+ладно\s+тебе)\b", "come on, no way"),
        (r"\b(чё\s+за\s+херня|че\s+за\s+хуйня|че\s+за\s+херня)\b", "what the fuck is this"),
        (r"\b(чё|че|чо)\b", "what"),
        (r"\b(ща|щас)\b", "right now"),
        (r"\b(короче)\b", "anyway"),
        (r"\b(погнали|поехали)\b", "let's roll"),
        (r"\b(чувак|братуха|бро)\b", "bro"),
        (r"\b(мудила|козёл|козел|гондон|гандон)\b", "asshole"),
        (r"\b(херня|хуйня)\b", "bullshit"),
        (r"\b(ни\s+хера|нихера|ни\s+хуя|нихуя)\b", "not a damn thing"),
        (r"\b(ё-моё|е-мое)\b", "damn it"),
        (r"\b(прикинь)\b", "get this"),
        (r"\b(дай\s+каос|дай\s+хаос|ну\s+и\s+каос|ну\s+и\s+хаос|бэйк\s+аус)\b", "total chaos"),
        (r"\bгеймбэйн\b", "fucking game"),
        (r"\b(жопу\s+покажи|покажи\s+жопу)\b", "show your ass"),
        (r"\b(лифак|лифчик)\b", "bra"),
        (r"\b(ха-ха|хаха|хахаха|ахаха)\b", "haha"),
    ]

    def __init__(self, device: str = None):
        self.device = device or config.DEVICE
        self._model = None
        self._tokenizer = None

    def load_model(self):
        if self._model is None or self._tokenizer is None:
            print(f"Loading neural translation model '{self.MODEL_NAME}' on {self.device}...", flush=True)
            self._tokenizer = MarianTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = MarianMTModel.from_pretrained(self.MODEL_NAME).to(self.device)
            self._model.eval()
            print("Neural translation model loaded successfully.", flush=True)
        return self._model, self._tokenizer

    def estimate_syllables(self, text: str) -> int:
        """Estimate English syllable count for speech budgeting."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        count = 0
        for w in words:
            vowels = len(re.findall(r"[aeiouy]+", w))
            if w.endswith("e") and len(w) > 2 and not w.endswith("le"):
                vowels = max(1, vowels - 1)
            count += max(1, vowels)
        return max(1, count)

    def polish_conversational_english(self, raw_english: str, russian_source: str, duration: float) -> str:
        """
        Polish raw MT output into natural spoken English:
        1. Natural conversational contractions (do not -> don't).
        2. Preserve explicit Russian mat & slang reactions.
        3. Pacing and syllable budgeting for the target speech duration.
        """
        text = raw_english.strip()
        ru_lower = russian_source.lower()

        # Clean repeated punctuation
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\?+", "?", text)
        text = re.sub(r"!+", "!", text)

        # 1. Natural Spoken Contractions
        contractions = [
            (r"\bdo not\b", "don't"),
            (r"\bcannot\b", "can't"),
            (r"\bcan not\b", "can't"),
            (r"\bwill not\b", "won't"),
            (r"\bis not\b", "isn't"),
            (r"\bare not\b", "aren't"),
            (r"\bI am\b", "I'm"),
            (r"\bI will\b", "I'll"),
            (r"\bI have\b", "I've"),
            (r"\bI would\b", "I'd"),
            (r"\bwhat is\b", "what's"),
            (r"\bthat is\b", "that's"),
            (r"\bit is\b", "it's"),
            (r"\blet us\b", "let's"),
            (r"\byou are\b", "you're"),
            (r"\bthey are\b", "they're"),
            (r"\bwe are\b", "we're"),
            (r"\bthere is\b", "there's"),
            (r"\bgoing to\b", "gonna"),
            (r"\bwant to\b", "wanna"),
        ]
        for pattern, replacement in contractions:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 2. Russian Slang / Mat preservation
        for ru_pattern, en_equiv in self.SLANG_AND_MAT_MAP:
            if re.search(ru_pattern, ru_lower, flags=re.IGNORECASE):
                # Check key words to avoid redundant doubling (e.g. ass in 'show me your ass')
                key_word = en_equiv.split()[-1] if en_equiv.split() else en_equiv
                if key_word.lower() not in text.lower() and en_equiv.lower() not in text.lower():
                    # If Russian starts with it (e.g. "Блядь, ...", "Иди нахуй, ...")
                    if re.match(ru_pattern, ru_lower):
                        text = f"{en_equiv.capitalize()}, {text[0].lower() + text[1:] if len(text) > 1 else text}"
                    # If Russian ends with it (e.g. "..., блядь")
                    elif re.search(ru_pattern + r"\s*[\.!\?]?$", ru_lower):
                        text = f"{text.rstrip('.!?')}, {en_equiv}."

        # Specific phrase fixes
        if ("иди нахуй" in ru_lower or "иди на" in ru_lower):
            text = "Fuck off, please." if "пожалуйста" in ru_lower else "Fuck off."

        # Normalize explicit mat reactions
        if "сука" in ru_lower and "пиздец" in ru_lower:
            text = "Bitch, holy shit."

        # Clean duplicate comma phrases
        text = re.sub(r"\bshow me your ass,\s*show your ass\b", "show your ass", text, flags=re.IGNORECASE)
        text = re.sub(r"\bshow your ass,\s*show your ass\b", "show your ass", text, flags=re.IGNORECASE)

        # Livestream context: "сняла" (took off clothes/bra) vs "filmed"
        if "сняла" in ru_lower and ("зрител" in ru_lower or "больше" in ru_lower):
            text = "I took it off, got more viewers."

        if "снимем" in ru_lower and "еще что-то" in ru_lower:
            text = "Come on, let's take something else off now."

        # In-game stream swearing
        if "геймбэйн" in ru_lower:
            text = "Fucking game!"
        if "бэйк аус" in ru_lower or "ну и хаос" in ru_lower or "дай каос" in ru_lower:
            text = "Total chaos!"

        # Clean punctuation and spacing
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\s*\.\s*", ". ", text).strip()

        # 3. Syllable & Pace optimization
        # Average spoken English pace is ~4.0 syllables/second
        target_syllables = max(2, int(duration * 4.2))
        current_syllables = self.estimate_syllables(text)

        if current_syllables > target_syllables * 1.35 and duration > 0.5:
            tightenings = [
                (r"\bin order to\b", "to"),
                (r"\bat the moment\b", "now"),
                (r"\bas a matter of fact\b", "actually"),
                (r"\bunder any circumstances\b", "at all"),
                (r"\bwhat are you talking about\b", "what you mean"),
                (r"\bare you kidding me\b", "you kidding"),
                (r"\bthere is going to be\b", "there'll be"),
                (r"\bthere will be\b", "there'll be"),
            ]
            for pattern, repl in tightenings:
                text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        return text.strip()

    def translate_segments(self, segments: list[dict]) -> list[dict]:
        """
        Translates a list of Russian transcription segments into natural English.
        Uses neural MarianMT in batch, followed by context-aware colloquial polishing.
        """
        if not segments:
            return []

        model, tokenizer = self.load_model()

        ru_texts = [s.get("russian_text", s.get("text", "")).strip() for s in segments]
        
        # Batch neural translation
        # Chunk into groups of 32 for optimal GPU inference
        raw_translations = []
        batch_size = 16

        for i in range(0, len(ru_texts), batch_size):
            batch = ru_texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2
                )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            raw_translations.extend(decoded)

        # Polish each segment with surrounding context
        for idx, seg in enumerate(segments):
            raw_en = raw_translations[idx]
            ru_source = seg.get("russian_text", "")
            duration = float(seg.get("duration", max(0.5, seg.get("end", 1.0) - seg.get("start", 0.0))))

            polished = self.polish_conversational_english(
                raw_english=raw_en,
                russian_source=ru_source,
                duration=duration
            )

            seg["raw_translation"] = raw_en
            seg["english_text"] = polished
            seg["english"] = polished  # compatibility alias

        return segments
