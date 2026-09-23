# FPL Team Picker

Predicts Fantasy Premier League points and turns the predictions into a weekly squad, lineup, transfer and chip recommendation for one manager.

## Language

**xP**:
A player's expected FPL points for a gameweek, or summed over the horizon.
_Avoid_: projected points, EV

**Recommendation**:
Everything the model tells the manager to do for one gameweek: squad, starting XI, formation, bench order, captain, vice-captain, transfers in/out and chip.
_Avoid_: decision, pick, output

**Policy**:
A rule that produces a gameweek choice during replay, such as hold, expected, multiperiod or oracle.
_Avoid_: strategy, bot

**Policy choice**:
What a Policy picked for one replayed gameweek: squad, starting XI and chip. It is not a Recommendation; nobody is told to act on it.
_Avoid_: decision

**Mode 1**:
A run that builds the full 15-man squad from scratch, ignoring the manager's current team.
_Avoid_: rebuild mode, draft

**Mode 2**:
A run that starts from the manager's current squad and recommends transfers against it.
_Avoid_: transfer mode, weekly mode
