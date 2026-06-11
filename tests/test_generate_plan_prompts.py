from datagen.swerebench_v2 import generate_harbor_tasks, generate_plan_prompts


def test_validate_plan_quality_warns_on_exact_patch_line():
    row = {
        "problem_statement": "Make account lookup return an empty string instead of \"stark\".",
        "interface": "Method: Account.getStarkName(address?: BigNumberish)",
        "patch": '+    return "stark" if result is None else normalize(result)\n',
    }
    plan = {
        "plan_steps": [
            'Change Account.getStarkName so it can return "stark" if result is None else normalize(result).'
        ],
        "preserved_requirements": [],
    }

    warnings = generate_plan_prompts.validate_plan_quality(row, plan)

    assert any(warning.startswith("copies_patch_line:") for warning in warnings)


def test_validate_plan_quality_checks_validation_notes_for_leakage():
    row = {
        "problem_statement": "Make account lookup return an empty string instead of \"stark\".",
        "interface": "Method: Account.getStarkName(address?: BigNumberish)",
        "patch": '+    return "stark" if result is None else normalize(result)\n',
    }
    plan = {
        "plan_steps": ["Trace Account.getStarkName through the no-name response path."],
        "preserved_requirements": [],
        "validation_notes": [
            'Verify return "stark" if result is None else normalize(result) in account.py.'
        ],
    }

    warnings = generate_plan_prompts.validate_plan_quality(row, plan)

    assert "contains_exact_file_path" in warnings
    assert any(warning.startswith("copies_patch_line:") for warning in warnings)


def test_validate_plan_quality_accepts_high_level_plan():
    row = {
        "problem_statement": "Make account lookup return an empty string instead of \"stark\".",
        "interface": "Method: Account.getStarkName(address?: BigNumberish)",
        "patch": '+    return "" if result is None else normalize(result)\n',
    }
    plan = {
        "plan_steps": [
            "Trace Account.getStarkName through the no-name response path and adjust its fallback behavior.",
            'Keep callers compatible while ensuring the no-result case returns an empty string, not "stark".',
        ],
        "preserved_requirements": [],
    }

    assert generate_plan_prompts.validate_plan_quality(row, plan) == []


def test_build_instruction_uses_planned_prompt():
    row = {
        "instance_id": "repo__project-1",
        "problem_statement": "Original task.",
        "interface": "",
    }
    plans = {
        "repo__project-1": {
            "rollout_prompt": "Original task.\n\nAdditional guidance:\n- Inspect behavior.\n",
            "sft_prompt": "Original task.\n",
        }
    }

    instruction = generate_harbor_tasks.build_instruction(row, "planned", plans=plans)

    assert "Additional guidance" in instruction
    assert instruction.endswith("\n")


def test_build_sft_instruction_uses_unhinted_planned_prompt():
    row = {
        "instance_id": "repo__project-1",
        "problem_statement": "Original task.",
        "interface": "",
    }
    plans = {
        "repo__project-1": {
            "rollout_prompt": "Original task.\n\nAdditional guidance:\n- Inspect behavior.\n",
            "sft_prompt": "Original task.\n",
        }
    }

    instruction = generate_harbor_tasks.build_sft_instruction(row, "planned", plans=plans)

    assert instruction == "Original task.\n"


def test_planned_prompt_keeps_original_task_primary_and_adds_guidance():
    row = {
        "problem_statement": "Fix the cache invalidation bug.",
    }
    plan = {
        "plan_steps": ["Inspect the cache lifecycle around update and delete operations."],
        "preserved_requirements": ["Preserve the public cache refresh behavior."],
        "validation_notes": ["Exercise update, delete, and repeated lookup scenarios."],
    }

    prompt = generate_plan_prompts.planned_prompt(row, plan)

    assert prompt.startswith("Fix the cache invalidation bug.")
    assert "Additional guidance:" in prompt
    assert "Compatibility requirements:" in prompt
    assert "Validation focus:" in prompt


def test_add_prompt_variants_keeps_rollout_hints_separate_from_sft_prompt():
    row = {
        "problem_statement": "Fix the cache invalidation bug.",
    }
    plan = {
        "plan_steps": ["Inspect the cache lifecycle around update and delete operations."],
        "preserved_requirements": [],
        "validation_notes": [],
    }

    generate_plan_prompts.add_prompt_variants(row, plan)

    assert plan["planned_prompt"] == plan["rollout_prompt"]
    assert plan["hinted_prompt"] == plan["rollout_prompt"]
    assert "Additional guidance:" in plan["rollout_prompt"]
    assert plan["sft_prompt"] == "Fix the cache invalidation bug.\n"
