"""What the robot says: the words for an action, looked up in config.PHRASES.

Kept apart from Act.py (which needs MediaPipe for drawing) so that the
simulation can use the same phrases.
"""

from coach import config


def phrase_for(action, cue, exercise, simplified):
    """Look up what the robot says for an action in config.PHRASES.

    Tries the most specific key first: cue + exercise (+ simplified), then the cue,
    then the exercise, then the general phrase of the action.
    """
    contexts = []
    if cue and exercise:
        if simplified:
            contexts.append(f'{cue}:{exercise}:simplified')
        contexts.append(f'{cue}:{exercise}')
    if cue:
        contexts.append(cue)
    if exercise:
        contexts.append(exercise)
    contexts.append(None)
    for context in contexts:
        phrase = config.PHRASES.get((action.name, context))
        if phrase:
            return phrase
    return action.name
