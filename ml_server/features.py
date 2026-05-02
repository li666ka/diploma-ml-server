# 4. NRC lexicon loader + emotional feature extraction
import pandas as pd
import re
import emoji
from collections import defaultdict
import os


def load_nrc_el(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["word", "emotion", "value"])
    df = df[df["value"] == 1]
    lex = defaultdict(list)
    for _, row in df.iterrows():
        lex[row["word"]].append(row["emotion"])
    return lex


def load_nrc_eil(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["word", "emotion", "intensity"])
    return {row["word"]: float(row["intensity"]) for _, row in df.iterrows()}

def extract_stylistic_features(text, feature_names):
    """
    Compute stylistic features from raw (untokenized, original-case) text.

    Returns dict with requested features only (those in feature_names).
    Supports: caps_ratio, ttr, repetition_score, avg_word_length
    """
    if not isinstance(text, str) or not text.strip():
        return {f: 0.0 for f in feature_names}

    result = {}

    # caps_ratio — частка великих літер серед усіх літер
    if "caps_ratio" in feature_names:
        letters = [c for c in text if c.isalpha()]
        if letters:
            caps = sum(1 for c in letters if c.isupper())
            result["caps_ratio"] = caps / len(letters)
        else:
            result["caps_ratio"] = 0.0

    # Tokenize words — lowercase для ttr/repetition/avg_word_length
    # Split по whitespace і видалити порожні + пунктуацію
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())

    if not words:
        for f in ("ttr", "repetition_score", "avg_word_length"):
            if f in feature_names:
                result[f] = 0.0
        return result

    # ttr — Type-Token Ratio (лексичне різноманіття)
    if "ttr" in feature_names:
        if len(words) < 10:
            # Для коротких текстів ttr нестабільний — використовуємо 1.0 як neutral
            result["ttr"] = 1.0
        else:
            unique = len(set(words))
            result["ttr"] = unique / len(words)

    # repetition_score — інверс ttr
    if "repetition_score" in feature_names:
        if len(words) < 10:
            result["repetition_score"] = 0.0
        else:
            unique = len(set(words))
            result["repetition_score"] = 1.0 - (unique / len(words))

    # avg_word_length — середня довжина слова
    if "avg_word_length" in feature_names:
        total_chars = sum(len(w) for w in words)
        result["avg_word_length"] = total_chars / len(words)

    return result


print("✓ Stylistic features: caps_ratio, ttr, repetition_score, avg_word_length")

# Define feature groups globally
EMOTIONAL_FEATURES = {
    "anger_score", "fear_score", "anticipation_score", "trust_score",
    "surprise_score", "sadness_score", "joy_score", "disgust_score",
    "positive_score", "negative_score",
    "sentiment_score", "emotion_intensity",
    "emoji_count", "exclamation_count",
}

STYLISTIC_FEATURES = {
    "caps_ratio", "ttr", "repetition_score", "avg_word_length",
}

RHETORICAL_FEATURES = {
    "clickbait_score", "authority_refs", "pronoun_ratio", "question_count",
}

SOCIAL_FEATURES = {
    # Profile counts (8)
    "followers_count_norm", "friends_count_norm", "ff_ratio",
    "statuses_count_norm", "favourites_count_norm", "listed_count_norm",
    "account_age_norm", "statuses_per_day",
    # Profile flags + strings (8)
    "verified", "geo_enabled", "has_profile",
    "has_description", "has_location",
    "description_length_norm", "screen_name_length_norm", "screen_name_digits_ratio",
    # Engagement (5)
    "like_count_norm", "retweet_count_norm", "reply_count_norm",
    "like_to_retweet_ratio", "engagement_rate",
}

# Keyword lists for rhetorical features
# Compiled regex patterns — performance-critical (called millions of times)
_CLICKBAIT_KEYWORDS = [
    r"\bshocking\b", r"\byou won\'?t believe\b", r"\bamazing\b",
    r"\bmind\s*blown\b", r"\bsecret\b", r"\bthis is why\b",
    r"\bwhat happened next\b", r"\bbreaking\b", r"\bunbelievable\b",
    r"\binsane\b", r"\bepic\b", r"\bjaw\s*dropping\b",
    r"\bwill blow your mind\b", r"\bthey don\'?t want you to know\b",
    r"\bdoctors hate\b", r"\bone weird trick\b", r"\bhorrifying\b",
    r"\bterrifying\b", r"\boutraged\b", r"\bbombshell\b",
    r"\bexposed\b", r"\brevealed\b", r"\bstunned\b",
]
_CLICKBAIT_PATTERN = re.compile("|".join(_CLICKBAIT_KEYWORDS), re.IGNORECASE)

_AUTHORITY_PATTERNS = [
    r"\bexperts? say\b", r"\bscientists? (?:say|claim|believe|warn)\b",
    r"\bstudies show\b", r"\bresearch shows\b", r"\bresearchers? (?:say|claim|found)\b",
    r"\bsources? (?:tell|say|claim|report)\b", r"\bofficials? (?:reveal|claim|say)\b",
    r"\binsiders? (?:report|say|reveal)\b", r"\banonymous source\b",
    r"\baccording to (?:experts?|sources?|insiders?|officials?)\b",
    r"\bauthorities (?:say|claim)\b", r"\banalysts? (?:say|predict|warn)\b",
    r"\bit (?:is )?reported that\b", r"\bsome (?:say|claim|believe)\b",
    r"\bcritics (?:say|claim)\b", r"\bpeople are saying\b",
]
_AUTHORITY_PATTERN = re.compile("|".join(_AUTHORITY_PATTERNS), re.IGNORECASE)

_GROUP_PRONOUNS = {"we", "us", "our", "ours", "they", "them", "their", "theirs"}


def extract_rhetorical_features(text, feature_names):
    """
    Compute rhetorical (manipulation) features from raw text.

    Supports: clickbait_score, authority_refs, pronoun_ratio, question_count
    """
    if not isinstance(text, str) or not text.strip():
        return {f: 0.0 for f in feature_names}

    result = {}

    # Tokenize for word-level features
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    total_words = max(len(words), 1)

    # clickbait_score — density of clickbait keywords
    if "clickbait_score" in feature_names:
        matches = _CLICKBAIT_PATTERN.findall(text)
        result["clickbait_score"] = len(matches) / total_words * 10  # scale up (rare feature)

    # authority_refs — vague/anonymous authority references
    if "authority_refs" in feature_names:
        matches = _AUTHORITY_PATTERN.findall(text)
        # Normalize by sentence count
        sentences = max(len(re.findall(r"[.!?]+", text)), 1)
        result["authority_refs"] = len(matches) / sentences

    # pronoun_ratio — us/them pronouns density (group polarization)
    if "pronoun_ratio" in feature_names:
        group_pronouns = sum(1 for w in words if w in _GROUP_PRONOUNS)
        result["pronoun_ratio"] = group_pronouns / total_words

    # question_count — question marks normalized
    if "question_count" in feature_names:
        result["question_count"] = text.count("?") / total_words * 100  # scale up

    return result


print("✓ Rhetorical features: clickbait_score, authority_refs, pronoun_ratio, question_count")


# ── Social features — computed from user profile row (not text) ─────
# These require joined tweets.csv ↔ users.csv (не з тексту)

# Normalization constants (tuned for FakeNewsNet distribution, not data-leaky)
# Stats from literature + Twitter population distribution
_LOG_DIVISOR = 20.0          # log1p(500M) ≈ 20 → sensible [0,1] range
_LOG_DIVISOR_LISTED = 15.0   # listed_count has smaller range
_FF_RATIO_CLIP = 100.0       # cap ratio at 100
_REFERENCE_YEAR_UTC = 1577836800.0  # 2020-01-01 UTC (FakeNewsNet cutoff era)
_MAX_ACCOUNT_AGE_YEARS = 15.0


def extract_social_features(user_row, feature_names, tweet_row=None):
    """
    Compute social features.

    Args:
        user_row: dict/Series з колонками users.csv (можна None)
        feature_names: list of features to compute
        tweet_row: dict/Series з tweet (для engagement features). Може бути None.

    Returns: dict {feature_name: float}
    """
    result = {f: 0.0 for f in feature_names}

    # ── Helpers ──
    def _num(row, col, default=0.0):
        if row is None:
            return default
        try:
            v = row.get(col) if hasattr(row, "get") else row[col]
            if v is None or pd.isna(v):
                return default
            return float(v)
        except (KeyError, ValueError, TypeError):
            return default

    def _str(row, col, default=""):
        if row is None:
            return default
        try:
            v = row.get(col) if hasattr(row, "get") else row[col]
            if v is None or pd.isna(v):
                return default
            return str(v)
        except (KeyError, TypeError):
            return default

    def _bool(row, col):
        if row is None:
            return 0.0
        try:
            v = row.get(col) if hasattr(row, "get") else row[col]
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            if isinstance(v, (int, float)) and not pd.isna(v):
                return 1.0 if v else 0.0
            if isinstance(v, str):
                return 1.0 if v.strip().lower() in ("true", "1", "t", "yes") else 0.0
            return 0.0
        except (KeyError, TypeError):
            return 0.0

    # ── Determine if real profile data exists ──
    followers = _num(user_row, "user_followers_count", -1) if user_row is not None else -1
    has_real_profile = followers >= 0

    if "has_profile" in feature_names:
        result["has_profile"] = 1.0 if has_real_profile else 0.0

    # ── Engagement features (від tweet_row, не залежать від user profile) ──
    likes = _num(tweet_row, "like_count", 0)
    retweets = _num(tweet_row, "retweet_count", 0)
    replies = _num(tweet_row, "reply_count", 0)

    if "like_count_norm" in feature_names:
        # log1p(450) ≈ 6.1; нормалізація на 7
        result["like_count_norm"] = min(math.log1p(likes) / 7.0, 1.0)

    if "retweet_count_norm" in feature_names:
        result["retweet_count_norm"] = min(math.log1p(retweets) / 7.0, 1.0)

    if "reply_count_norm" in feature_names:
        result["reply_count_norm"] = min(math.log1p(replies) / 7.0, 1.0)

    if "like_to_retweet_ratio" in feature_names:
        # Shu 2020 ключова фіча: real news має higher likes/retweets ratio
        # Capped at 10, normalized by 10 → ∈[0, 1]
        ratio = likes / (retweets + 1.0)
        result["like_to_retweet_ratio"] = min(ratio, 10.0) / 10.0

    if "engagement_rate" in feature_names:
        # (likes+retweets+replies) / (followers+1) — Cha 2010 metric
        total_engagement = likes + retweets + replies
        followers_for_rate = max(followers, 0)  # якщо нема profile → 0
        rate = total_engagement / (followers_for_rate + 1.0)
        # Most engagement_rate < 1.0 у реальних даних, capping at 10 for safety
        result["engagement_rate"] = min(rate, 10.0) / 10.0

    # ── Profile features (потрібен user_row) ──
    if not has_real_profile:
        # Без profile — всі profile features = 0 (вже initialized)
        return result

    friends = _num(user_row, "user_friends_count")
    statuses = _num(user_row, "user_statuses_count")
    favourites = _num(user_row, "user_favourites_count")
    listed = _num(user_row, "user_listed_count")
    created_at = _num(user_row, "user_created_at")
    description = _str(user_row, "user_description")
    location = _str(user_row, "user_location")
    screen_name = _str(user_row, "user_screen_name")

    # Counts
    if "followers_count_norm" in feature_names:
        result["followers_count_norm"] = min(math.log1p(followers) / 20.0, 1.0)

    if "friends_count_norm" in feature_names:
        result["friends_count_norm"] = min(math.log1p(friends) / 20.0, 1.0)

    if "ff_ratio" in feature_names:
        ratio = followers / (friends + 1.0)
        result["ff_ratio"] = min(ratio, 100.0) / 100.0

    if "statuses_count_norm" in feature_names:
        result["statuses_count_norm"] = min(math.log1p(statuses) / 20.0, 1.0)

    if "favourites_count_norm" in feature_names:
        result["favourites_count_norm"] = min(math.log1p(favourites) / 20.0, 1.0)

    if "listed_count_norm" in feature_names:
        result["listed_count_norm"] = min(math.log1p(listed) / 15.0, 1.0)

    # Account age + activity rate
    REFERENCE_UNIX = 1577836800.0  # 2020-01-01
    age_seconds = max(REFERENCE_UNIX - created_at, 0.0) if created_at > 0 else 0.0
    age_days = age_seconds / 86400.0
    age_years = age_days / 365.25

    if "account_age_norm" in feature_names:
        result["account_age_norm"] = min(age_years / 15.0, 1.0)

    if "statuses_per_day" in feature_names:
        # Yang 2020 SHAP top-5 — bot signal (high tweet_freq → bot)
        if age_days > 1.0:
            rate_per_day = statuses / age_days
            # Typical: humans 1-50/day, bots 100+/day. log1p(1000) ≈ 6.9 → /7 cap
            result["statuses_per_day"] = min(math.log1p(rate_per_day) / 7.0, 1.0)
        else:
            result["statuses_per_day"] = 0.0

    # Booleans
    if "verified" in feature_names:
        result["verified"] = _bool(user_row, "user_verified")

    if "geo_enabled" in feature_names:
        result["geo_enabled"] = _bool(user_row, "user_geo_enabled")

    if "has_description" in feature_names:
        result["has_description"] = 1.0 if description.strip() else 0.0

    if "has_location" in feature_names:
        result["has_location"] = 1.0 if location.strip() else 0.0

    # Strings
    if "description_length_norm" in feature_names:
        # Twitter bio max 160 chars
        result["description_length_norm"] = min(len(description) / 160.0, 1.0)

    if "screen_name_length_norm" in feature_names:
        # Twitter handle max 15 chars (без @)
        result["screen_name_length_norm"] = min(len(screen_name) / 15.0, 1.0)

    if "screen_name_digits_ratio" in feature_names:
        # Beskow 2019: bots часто мають digit-heavy handles
        if screen_name:
            digits = sum(1 for c in screen_name if c.isdigit())
            result["screen_name_digits_ratio"] = digits / len(screen_name)
        else:
            result["screen_name_digits_ratio"] = 0.0

    return result


import math  # required by extract_social_features
print("✓ Social features: 21 фіч (8 counts + 8 flags/strings + 5 engagement)")

def _extract_emotional(text, nrc_el, nrc_eil, feature_list):
    if not isinstance(text, str) or not text.strip():
        return {f: 0.0 for f in feature_list}

    words = re.findall(r"\w+", text.lower())
    total_words = len(words) if words else 1

    emotion_counts = defaultdict(int)
    for word in words:
        if word in nrc_el:
            for emo in nrc_el[word]:
                emotion_counts[emo] += 1

    intensity_sum = 0
    count_eil = 0
    for word in words:
        if word in nrc_eil:
            intensity_sum += nrc_eil[word]
            count_eil += 1

    base = {
        "anger_score":         emotion_counts["anger"] / total_words,
        "fear_score":          emotion_counts["fear"] / total_words,
        "anticipation_score":  emotion_counts["anticipation"] / total_words,
        "trust_score":         emotion_counts["trust"] / total_words,
        "surprise_score":      emotion_counts["surprise"] / total_words,
        "sadness_score":       emotion_counts["sadness"] / total_words,
        "joy_score":           emotion_counts["joy"] / total_words,
        "disgust_score":       emotion_counts["disgust"] / total_words,
        "positive_score":      emotion_counts["positive"] / total_words,
        "negative_score":      emotion_counts["negative"] / total_words,
    }

    results = {}
    for f in feature_list:
        if f in base:
            results[f] = base[f]
        elif f == "sentiment_score":
            results[f] = (emotion_counts["positive"] - emotion_counts["negative"]) / total_words
        elif f == "emotion_intensity":
            results[f] = intensity_sum / count_eil if count_eil > 0 else 0
        elif f == "emoji_count":
            results[f] = float(len(emoji.emoji_list(text)))
        elif f == "exclamation_count":
            results[f] = float(text.count("!"))
    return results


def extract_features(text, nrc_el, nrc_eil, feature_names, user_row=None):
    """
    Dispatch features to correct computation function.

    Routes by feature name:
    - Emotional (14)  → NRC lookup via _extract_emotional (from text)
    - Stylistic (4)   → stylistic computation (from text)
    - Rhetorical (4)  → manipulation patterns (from text)
    - Social (9)      → from user_row (NOT from text) — required user profile

    Args:
        text: the tweet text (for text-based features)
        user_row: user profile row (dict/Series) for social features (optional)
    """
    emo_feats = [f for f in feature_names if f in EMOTIONAL_FEATURES]
    styl_feats = [f for f in feature_names if f in STYLISTIC_FEATURES]
    rhet_feats = [f for f in feature_names if f in RHETORICAL_FEATURES]
    soc_feats = [f for f in feature_names if f in SOCIAL_FEATURES]

    result = {}

    if emo_feats:
        result.update(_extract_emotional(text, nrc_el, nrc_eil, emo_feats))

    if styl_feats:
        result.update(extract_stylistic_features(text, styl_feats))

    if rhet_feats:
        result.update(extract_rhetorical_features(text, rhet_feats))

    if soc_feats:
        # tweet_row passed via user_row's parent context (handled by caller)
        # extract_features signature stays same — caller may include tweet_row in user_row
        result.update(extract_social_features(user_row, soc_feats, tweet_row=user_row))

    # Fill missing (if caller asked for something unknown)
    for f in feature_names:
        if f not in result:
            result[f] = 0.0

    return result


try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    nrc_el_path = os.path.join(_THIS_DIR, "NRC", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
    nrc_eil_path = os.path.join(_THIS_DIR, "NRC", "NRC-Emotion-Intensity-Lexicon-v1.txt")
    nrc_el = load_nrc_el(nrc_el_path)
    nrc_eil = load_nrc_eil(nrc_eil_path)
    if not nrc_el or not nrc_eil:
        raise ValueError("NRC lexicon empty")
    print(f"✓ NRC-EL: {len(nrc_el):,} words")
    print(f"✓ NRC-EIL: {len(nrc_eil):,} words")
except Exception as e:
    print(f"NRC ERROR: {e}")