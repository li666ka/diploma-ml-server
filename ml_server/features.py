# 4. NRC lexicon loader + emotional feature extraction
import math
import os
import re
from collections import defaultdict

import emoji
import pandas as pd


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
    # Stylistic (form)
    "caps_ratio", "ttr", "repetition_score", "avg_word_length",
    # Rhetorical (manipulation patterns) — раніше окрема група
    "clickbait_score", "authority_refs", "pronoun_ratio", "question_count",
}

SOCIAL_FEATURES = {
    # Profile counts (5)
    "followers_count_norm", "friends_count_norm", "ff_ratio",
    "statuses_count_norm",
    "account_age_norm", "statuses_per_day",
    # Profile flags + strings (5)
    "verified",
    "has_description", "has_location",
    "description_length_norm", "screen_name_length_norm", "screen_name_digits_ratio",
    # Engagement (5)
    "like_count_norm", "retweet_count_norm", "reply_count_norm",
    "like_to_retweet_ratio", "engagement_rate",
    # Graph cascade (6) — обчислюються з tweets/retweets/replies для article
    "cascade_depth_norm", "cascade_breadth_norm", "lifetime_hours_norm",
    "retweets_per_tweet", "replies_per_tweet", "unique_users_norm",
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


def extract_social_features(
    user_row, feature_names, tweet_row=None,
    tweet_engagement_lookup=None,
):
    """
    Compute social features.

    Args:
        user_row: dict/Series з колонками users.csv (можна None)
        feature_names: list of features to compute
        tweet_row: dict/Series з tweet (для engagement features). Може бути None.
        tweet_engagement_lookup: dict {tweet_id_str: {"retweets": int, "replies": int}}.
            Якщо None — retweets/replies вважаються 0 (tweets.csv не містить
            колонок retweet_count/reply_count, тож їх треба обчислювати
            заздалегідь з retweets.csv/replies.csv).

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

    # ── Engagement features (від tweet_row, не залежать від user profile) ──
    likes = _num(tweet_row, "like_count", 0)
    retweets = 0
    replies = 0
    if tweet_engagement_lookup is not None and tweet_row is not None:
        try:
            tid_raw = (
                tweet_row.get("tweet_id", "")
                if hasattr(tweet_row, "get")
                else tweet_row["tweet_id"]
            )
        except (KeyError, TypeError):
            tid_raw = ""
        tid = str(tid_raw) if tid_raw is not None else ""
        eng = tweet_engagement_lookup.get(tid)
        if eng:
            retweets = eng.get("retweets", 0)
            replies = eng.get("replies", 0)

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


print("✓ Social features: 23 фіч (6 counts + 6 flags/strings + 5 engagement + 6 graph)")


# ── Graph cascade features (per-article) ──────────────────────────────────

def extract_graph_features(
    article_id,
    tweets_df,       # subset де article_id == this article
    retweets_df,     # subset де original_tweet_id IN tweets_df.tweet_id
    replies_df,      # subset де parent_tweet_id IN tweets_df.tweet_id (transitive)
    feature_names,
):
    """Compute graph cascade features for one article.

    Очікує df-и які вже відфільтровані для цієї статті.
    """
    result = {f: 0.0 for f in feature_names}

    n_tweets = len(tweets_df)
    n_retweets = len(retweets_df)
    n_replies = len(replies_df)

    if n_tweets == 0:
        return result  # Без твітів — все 0

    # ── retweets_per_tweet, replies_per_tweet ──
    if "retweets_per_tweet" in feature_names:
        ratio = n_retweets / max(n_tweets, 1)
        result["retweets_per_tweet"] = min(ratio, 50.0) / 50.0

    if "replies_per_tweet" in feature_names:
        ratio = n_replies / max(n_tweets, 1)
        result["replies_per_tweet"] = min(ratio, 50.0) / 50.0

    # ── unique_users_norm ──
    if "unique_users_norm" in feature_names:
        users = set()
        if "user_id" in tweets_df.columns:
            users.update(tweets_df["user_id"].dropna().astype(str).tolist())
        if "user_id" in retweets_df.columns:
            users.update(retweets_df["user_id"].dropna().astype(str).tolist())
        if "user_id" in replies_df.columns:
            users.update(replies_df["user_id"].dropna().astype(str).tolist())
        result["unique_users_norm"] = min(math.log1p(len(users)) / 10.0, 1.0)

    # ── lifetime_hours_norm ──
    if "lifetime_hours_norm" in feature_names:
        timestamps = []
        for df, col in [
            (tweets_df, "tweet_created_at"),
            (retweets_df, "retweet_created_at"),
            (replies_df, "reply_created_at"),
        ]:
            if col not in df.columns:
                continue
            raw = df[col].dropna()
            if len(raw) == 0:
                continue
            # Try numeric first (Unix seconds)
            ts_numeric = pd.to_numeric(raw, errors="coerce")
            # Для не-числових значень — пробуємо парсити як datetime string
            # (Twitter API: "Sun May 25 08:06:04 +0000 2014").
            mask_failed = ts_numeric.isna()
            if mask_failed.any():
                try:
                    ts_datetime = pd.to_datetime(
                        raw[mask_failed], errors="coerce", utc=True,
                    )
                    valid_dt = ts_datetime.dropna()
                    if len(valid_dt) > 0:
                        ts_unix = (
                            valid_dt.astype("int64") // 10**9
                        ).astype("float64")
                        ts_numeric.loc[ts_unix.index] = ts_unix
                except (ValueError, TypeError):
                    pass
            ts_numeric = ts_numeric.dropna()
            timestamps.extend(ts_numeric.tolist())
        if len(timestamps) >= 2:
            lifetime_seconds = max(timestamps) - min(timestamps)
            lifetime_hours = lifetime_seconds / 3600.0
            # 30 days = 720h як максимум
            result["lifetime_hours_norm"] = min(lifetime_hours / 720.0, 1.0)

    # ── cascade_depth_norm: BFS глибина reply tree ──
    # ── cascade_breadth_norm: max ширина на одному рівні ──
    if "cascade_depth_norm" in feature_names or "cascade_breadth_norm" in feature_names:
        depth, breadth = _compute_cascade_topology(
            tweets_df, replies_df, max_depth=10
        )
        if "cascade_depth_norm" in feature_names:
            result["cascade_depth_norm"] = min(depth / 10.0, 1.0)
        if "cascade_breadth_norm" in feature_names:
            result["cascade_breadth_norm"] = min(breadth / 20.0, 1.0)

    return result


def _compute_cascade_topology(tweets_df, replies_df, max_depth=10):
    """Returns (max_depth, max_breadth) reply tree."""
    if len(tweets_df) == 0:
        return 0, 0

    # Tweets — рівень 0 (корені). max_breadth вимірює РОЗШИРЕННЯ дерева
    # replies (max replies per level), а не кількість коренів.
    current_level_ids = set(tweets_df["tweet_id"].astype(str).tolist())
    max_depth_reached = 0
    max_breadth = 0

    if len(replies_df) == 0 or "parent_tweet_id" not in replies_df.columns:
        return max_depth_reached, max_breadth

    # Build children lookup
    replies_df_clean = replies_df.copy()
    replies_df_clean["parent_tweet_id"] = replies_df_clean["parent_tweet_id"].astype(str)
    replies_df_clean["reply_id"] = replies_df_clean["reply_id"].astype(str)

    children_of = {}
    for parent_id, group in replies_df_clean.groupby("parent_tweet_id"):
        children_of[parent_id] = group["reply_id"].tolist()

    visited = set(current_level_ids)

    for depth in range(1, max_depth + 1):
        next_level = []
        for parent in current_level_ids:
            kids = children_of.get(parent, [])
            for kid in kids:
                if kid not in visited:
                    visited.add(kid)
                    next_level.append(kid)
        if not next_level:
            break
        max_depth_reached = depth
        max_breadth = max(max_breadth, len(next_level))
        current_level_ids = set(next_level)

    return max_depth_reached, max_breadth


print("✓ Graph features: cascade_depth_norm, cascade_breadth_norm, lifetime_hours_norm, "
      "retweets_per_tweet, replies_per_tweet, unique_users_norm")

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


def extract_features(
    text, nrc_el, nrc_eil, feature_names,
    user_row=None,
    tweet_row=None,
    article_tweets_df=None,
    article_retweets_df=None,
    article_replies_df=None,
    article_id=None,
    tweet_engagement_lookup=None,
):
    """
    Dispatch features to correct computation function.

    Routes by feature name:
    - Emotional (14)  → NRC lookup via _extract_emotional (from text)
    - Stylistic (8)   → stylistic + rhetorical (from text)
    - Social  (13)    → from user_row + tweet_row (NOT from text)
    - Graph    (6)    → from article_*_df (cascade topology)
    """
    emo_feats = [f for f in feature_names if f in EMOTIONAL_FEATURES]
    # styl тепер містить і чисто стилістичні, і риторичні
    styl_pure = {"caps_ratio", "ttr", "repetition_score", "avg_word_length"}
    rhet_pure = {"clickbait_score", "authority_refs", "pronoun_ratio", "question_count"}
    graph_pure = {
        "cascade_depth_norm", "cascade_breadth_norm", "lifetime_hours_norm",
        "retweets_per_tweet", "replies_per_tweet", "unique_users_norm",
    }
    styl_feats_pure = [f for f in feature_names if f in styl_pure]
    rhet_feats_pure = [f for f in feature_names if f in rhet_pure]
    graph_feats = [f for f in feature_names if f in graph_pure]
    soc_feats = [f for f in feature_names if f in SOCIAL_FEATURES and f not in graph_pure]

    result = {}

    if emo_feats:
        result.update(_extract_emotional(text, nrc_el, nrc_eil, emo_feats))

    if styl_feats_pure:
        result.update(extract_stylistic_features(text, styl_feats_pure))

    if rhet_feats_pure:
        result.update(extract_rhetorical_features(text, rhet_feats_pure))

    if soc_feats:
        result.update(extract_social_features(
            user_row, soc_feats, tweet_row=tweet_row,
            tweet_engagement_lookup=tweet_engagement_lookup,
        ))

    if graph_feats and article_tweets_df is not None:
        result.update(extract_graph_features(
            article_id, article_tweets_df, article_retweets_df,
            article_replies_df, graph_feats,
        ))

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