from scripts import rewrite_prompts


def test_extract_public_symbols_from_interface_declarations():
    interface = """\
Method: Account.getStarkName(address?: BigNumberish, StarknetIdContract?: string)
Location: src/account/default.ts
Function: SteppingPlayer(moves)
Inputs:
- moves - a deterministic description
"""

    assert rewrite_prompts.extract_public_symbols(interface) == [
        "Account.getStarkName",
        "SteppingPlayer",
    ]


def test_validate_rewrite_warns_when_public_symbol_is_lost():
    row = {
        "problem_statement": "pytest warns about our TestPlayer. Maybe rename it?",
        "interface": "Function: SteppingPlayer(moves)",
    }
    rewrite = {
        "rewritten_prompt": "Make the deterministic player avoid pytest collection warnings.",
        "preserved_requirements": [],
    }

    assert rewrite_prompts.validate_rewrite_quality(row, rewrite) == [
        "missing_public_symbol:SteppingPlayer"
    ]


def test_validate_rewrite_accepts_preserved_public_symbol():
    row = {
        "problem_statement": "pytest warns about our TestPlayer. Maybe rename it?",
        "interface": "Function: SteppingPlayer(moves)",
    }
    rewrite = {
        "rewritten_prompt": (
            "Rename the deterministic player to `SteppingPlayer` and keep it importable "
            "for existing SimpleTeam and GameMaster callers."
        ),
        "preserved_requirements": [],
    }

    assert rewrite_prompts.validate_rewrite_quality(row, rewrite) == []


def test_validate_rewrite_warns_when_exact_bug_literal_is_lost():
    row = {
        "problem_statement": (
            'Currently account.getStarkName() returns "stark" when an address has no '
            "Starknet name. Expected behavior: return an empty string."
        ),
        "interface": "Method: Account.getStarkName(address?: BigNumberish)",
    }
    rewrite = {
        "rewritten_prompt": (
            "Account.getStarkName() should return an empty string when the queried "
            "address has no Starknet name."
        ),
        "preserved_requirements": [],
    }

    assert rewrite_prompts.validate_rewrite_quality(row, rewrite) == [
        "missing_edge_literal:stark"
    ]


def test_validate_rewrite_accepts_exact_bug_literal():
    row = {
        "problem_statement": (
            'Currently account.getStarkName() returns "stark" when an address has no '
            "Starknet name. Expected behavior: return an empty string."
        ),
        "interface": "Method: Account.getStarkName(address?: BigNumberish)",
    }
    rewrite = {
        "rewritten_prompt": (
            "Account.getStarkName() should return an empty string, not `stark`, when "
            "the queried address has no Starknet name."
        ),
        "preserved_requirements": [],
    }

    assert rewrite_prompts.validate_rewrite_quality(row, rewrite) == []
