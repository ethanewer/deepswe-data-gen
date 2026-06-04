# aspp__pelita-412

- repo: ASPP/pelita
- language: python
- difficulty: medium

## Rewritten Prompt

pytest is warning that a class named `TestPlayer` is being collected as a test class even though it has an `__init__` method. Make the deterministic player usable without triggering pytest collection warnings, while keeping its behavior as a simple player that returns a predefined sequence of moves.

The player should be constructible with a sequence of move callables, a sequence of coordinate tuples, or a string of direction symbols. It must return those moves one after another when asked to act, and raise `ValueError` when the supplied input cannot be interpreted as a valid move sequence.

It should remain suitable for use with `SimpleTeam` and `GameMaster`.

## Preserved Requirements

- A pytest warning about collecting a class named `TestPlayer` with an `__init__` method must be eliminated.
- The deterministic player must still behave as a simple player that returns a predefined sequence of moves.
- The player must accept move callables, coordinate tuples, or a string of direction symbols as input.
- The player must yield the predefined moves one after another on each turn.
- The player must raise `ValueError` for invalid move sequences.
- The player must remain usable with `SimpleTeam` and `GameMaster`.

## Removed Noise

- Pytest warning code and file path details.
- Suggestion to rename the class as a direct hint.
- Issue template / benchmark framing.
- Reference to generated interface notes as metadata.
- Exact class/location wording from the notes.

## Risk Notes

- The original prompt only explicitly mentions a pytest collection warning; the exact public API name change is inferred from the interface notes.
- The requirement to keep the player usable with existing game components is preserved, but the precise class name is not stated in the original issue text.

## Original Prompt

pytest warns about our TestPlayer
    WC1 /tmp/group1/test/test_drunk_player.py cannot collect test class 'TestPlayer' because it has a __init__ constructor

Maybe rename it?

## Original Interface

Function: SteppingPlayer(moves)
Location: pelita/player/base.py (class SteppingPlayer, renamed from TestPlayer)
Inputs: 
- **moves** – a deterministic description of the player’s actions; accepted forms are a list/tuple of move callables (e.g., east, west), a list/tuple of coordinate tuples, or a string of direction symbols (e.g., “>><”). The parameter determines the sequence of moves the player will return on each turn.
Outputs: 
- Returns a new SteppingPlayer instance (subclass of AbstractPlayer) ready to be supplied to SimpleTeam and GameMaster. 
- Raises ValueError if the supplied moves cannot be interpreted as a valid move sequence.
Description: 
SteppingPlayer is a deterministic player used in tests; it yields the predefined moves one after another each time the game engine asks for a move. It replaces the former TestPlayer class and is constructed directly in the test suite (e.g., SteppingPlayer([]), SteppingPlayer('>-v>>>'), SteppingPlayer([(0,0)])).
