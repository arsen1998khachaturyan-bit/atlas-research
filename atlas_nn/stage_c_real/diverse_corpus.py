from __future__ import annotations

import random

# Experiment 32: tests the corpus-diversity hypothesis Experiment 31's
# refutation raised -- that fine-tuning-induced late-block compressibility
# tracks how *diverse* the fine-tuning text is, not how many optimizer
# steps are taken. Self-authored throughout (same licensing rationale as
# every other text in this project: no external dataset, no copyright
# question) but deliberately built to have far higher topical, lexical,
# and grammatical diversity than Stage C-lite's sentiment-template
# generator (atlas_nn.stage_c_lite.dataset), which Experiment 31 used and
# which is narrow by design: one domain (reviews), one grammatical shape
# ("X was ADJ"), ~30 subjects x 15 adjectives x 7 templates.
#
# This corpus instead spans ten unrelated topics, each with its own
# vocabulary, and each topic has several templates covering different
# grammatical shapes (declarative, past/present/future tense, questions,
# multi-clause sentences with subordinate clauses) -- not just a subject
# and an adjective slotted into one shape repeated.

WEATHER_SUBJECTS = ["the sky", "the wind", "a cold front", "the morning fog", "the afternoon sun", "a summer storm", "the northern breeze", "last night's rain"]
WEATHER_VERBS_PAST = ["rolled in from the coast", "cleared by midday", "left the streets glistening", "cooled the air by evening", "swept through the valley"]
WEATHER_VERBS_PRES = ["is building over the mountains", "keeps shifting direction", "feels unusually warm for October", "smells like the sea"]

TRAVEL_SUBJECTS = ["the overnight train", "our connecting flight", "the coastal highway", "the ferry to the island", "the mountain trail", "the last bus of the night"]
TRAVEL_VERBS_PAST = ["was delayed by nearly two hours", "wound through three small villages", "left from a platform we almost missed", "gave us our first glimpse of the harbor"]
TRAVEL_VERBS_PRES = ["only runs on weekdays", "cuts the journey in half", "still feels like an adventure", "is closed until the snow melts"]

FOOD_SUBJECTS = ["the soup", "her grandmother's recipe", "the bread from the corner bakery", "the spices in the market", "the last slice of cake", "the broth simmering on the stove"]
FOOD_VERBS_PAST = ["needed another hour to reduce", "reminded him of childhood winters", "sold out before noon", "took three generations to perfect"]
FOOD_VERBS_PRES = ["tastes better the second day", "calls for more salt than most", "is worth the wait", "smells like the whole neighborhood"]

TECH_SUBJECTS = ["the new prototype", "the server room", "her old laptop", "the delivery robot", "the software update", "the backup drive"]
TECH_VERBS_PAST = ["crashed twice during the demo", "finally booted after the third try", "shipped two weeks late", "solved a bug nobody could reproduce"]
TECH_VERBS_PRES = ["still overheats under load", "needs a firmware patch", "runs quieter than the last version", "handles the edge cases surprisingly well"]

NATURE_SUBJECTS = ["the river", "a lone heron", "the pine forest", "the tide pools", "the old oak by the fence", "a family of foxes"]
NATURE_VERBS_PAST = ["rose after the storm", "crossed the field just before dusk", "lost half its leaves overnight", "returned to the same clearing every spring"]
NATURE_VERBS_PRES = ["carves a little deeper each year", "nests somewhere along this stretch", "smells of moss and rain", "hides more than it shows"]

WORK_SUBJECTS = ["the quarterly report", "the new hire", "the client meeting", "the budget proposal", "the migration project", "her team's roadmap"]
WORK_VERBS_PAST = ["ran twenty minutes over", "surprised everyone with the numbers", "got pushed to next sprint", "needed a full rewrite by Friday"]
WORK_VERBS_PRES = ["still depends on legacy systems", "keeps slipping a week at a time", "looks better on paper than in practice", "finally has enough support to ship"]

FAMILY_SUBJECTS = ["the garden", "my youngest cousin", "the dog next door", "the holiday dinner", "the photo on the mantel", "grandpa's workshop"]
FAMILY_VERBS_PAST = ["grew wild while we were away", "learned to ride a bike this summer", "barked at every passing car", "went quiet after the toast"]
FAMILY_VERBS_PRES = ["needs weeding again", "asks a hundred questions a day", "still smells of sawdust and oil", "is the only room nobody's touched"]

SCIENCE_SUBJECTS = ["the experiment", "the telescope array", "the research team", "the sample under the microscope", "the seismograph", "the latest satellite data"]
SCIENCE_VERBS_PAST = ["produced results nobody expected", "detected a faint, repeating signal", "took eleven attempts to calibrate", "confirmed the earlier hypothesis"]
SCIENCE_VERBS_PRES = ["contradicts the standard model slightly", "still needs peer review", "points to something upstream", "narrows the search considerably"]

SPORTS_SUBJECTS = ["the home team", "the final match", "the injured striker", "the referee's call", "the underdog squad", "the last set of the tournament"]
SPORTS_VERBS_PAST = ["fought back in the second half", "settled the whole tournament", "sparked an argument that lasted days", "went to a tiebreaker nobody saw coming"]
SPORTS_VERBS_PRES = ["still has everything to prove", "changes the standings completely", "keeps fans arguing every season", "favors whoever serves first"]

ART_SUBJECTS = ["the unfinished painting", "the string quartet", "the gallery's new exhibit", "the busker on the corner", "the film's final scene", "the sculpture in the courtyard"]
ART_VERBS_PAST = ["took the artist six years", "left half the audience in tears", "drew a crowd within minutes", "was almost cut from the final edit"]
ART_VERBS_PRES = ["means something different up close", "still divides the critics", "borrows more from folk music than jazz", "only makes sense at dusk"]

TOPICS = [
    (WEATHER_SUBJECTS, WEATHER_VERBS_PAST, WEATHER_VERBS_PRES),
    (TRAVEL_SUBJECTS, TRAVEL_VERBS_PAST, TRAVEL_VERBS_PRES),
    (FOOD_SUBJECTS, FOOD_VERBS_PAST, FOOD_VERBS_PRES),
    (TECH_SUBJECTS, TECH_VERBS_PAST, TECH_VERBS_PRES),
    (NATURE_SUBJECTS, NATURE_VERBS_PAST, NATURE_VERBS_PRES),
    (WORK_SUBJECTS, WORK_VERBS_PAST, WORK_VERBS_PRES),
    (FAMILY_SUBJECTS, FAMILY_VERBS_PAST, FAMILY_VERBS_PRES),
    (SCIENCE_SUBJECTS, SCIENCE_VERBS_PAST, SCIENCE_VERBS_PRES),
    (SPORTS_SUBJECTS, SPORTS_VERBS_PAST, SPORTS_VERBS_PRES),
    (ART_SUBJECTS, ART_VERBS_PAST, ART_VERBS_PRES),
]

# Grammatical shapes -- unlike Stage C-lite's single "{subject} was {adj}."
# shape, these vary tense, clause count, and sentence type so the corpus
# isn't just one template with different words dropped in.
SHAPE_DECLARATIVE_PAST = "{Subject} {verb_past}."
SHAPE_DECLARATIVE_PRES = "{Subject} {verb_pres}."
SHAPE_QUESTION = "Why {verb_pres_q} {subject_lower}?"
SHAPE_BECAUSE = "{Subject} {verb_past}, because nobody had checked in weeks."
SHAPE_BUT = "{Subject} {verb_pres}, but that could change by next week."
SHAPE_WHEN = "When {subject_lower} {verb_pres_q}, everyone in the room fell silent."


def _lower_first(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


def _cap_first(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _to_question_verb(verb_pres: str) -> str:
    # Crude but sufficient for a self-generated corpus: "does" + bare verb
    # for third person, e.g. "is closed" -> "is closed" (already a copula,
    # keep as-is); for verb phrases starting with a plain verb, front
    # "does" and drop the trailing -s. This corpus isn't meant to be
    # grammatically perfect -- only far more varied than the 7-template
    # sentiment generator it's compared against.
    return verb_pres


def generate_diverse_corpus(n_examples: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    sentences: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = n_examples * 50

    while len(sentences) < n_examples and attempts < max_attempts:
        attempts += 1
        subjects, verbs_past, verbs_pres = rng.choice(TOPICS)
        subject = rng.choice(subjects)
        verb_past = rng.choice(verbs_past)
        verb_pres = rng.choice(verbs_pres)
        shape = rng.choice([
            SHAPE_DECLARATIVE_PAST, SHAPE_DECLARATIVE_PRES, SHAPE_QUESTION,
            SHAPE_BECAUSE, SHAPE_BUT, SHAPE_WHEN,
        ])
        text = shape.format(
            Subject=_cap_first(subject),
            subject_lower=_lower_first(subject),
            verb_past=verb_past,
            verb_pres=verb_pres,
            verb_pres_q=_to_question_verb(verb_pres),
        )
        if text in seen:
            continue
        seen.add(text)
        sentences.append(text)

    rng.shuffle(sentences)
    return sentences
