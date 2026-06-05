# 3yourmind__django-migration-linter-113

- repo: 3YOURMIND/django-migration-linter
- language: python
- difficulty: easy

## Rewritten Prompt

There is a bug in the migration linter CLI: when a migration-list file is provided with `--include-migration-from`, the linter should restrict linting to the migrations named in that file, but instead it ignores the file and runs against all migrations.

The migration-list reader must keep its public behavior: passing no path or any falsy value means “no file specified” and should return `None`; a missing or unreadable file should raise a generic `Exception` wrapping the underlying `IOError`; and a readable file should return a list of `(app_label, migration_name)` tuples, with an empty list meaning the file contained no valid migration entries. The `MigrationLinter.read_migrations_list(cls, migrations_file_path)` class method must remain available with that callable contract.

Fix the CLI behavior so that providing a migration-list file actually limits linting to the migrations listed there, instead of treating the input as if no file restriction was given.

## Preserved Requirements

- Using `--include-migration-from` with a migration-list file must restrict linting to the migrations named in that file.
- The linter must not ignore the provided migration-list file and fall back to linting all migrations.
- MigrationLinter.read_migrations_list(cls, migrations_file_path) must remain available.
- Passing a falsy value for `migrations_file_path` means no file is specified and must return `None`.
- A missing or unreadable migrations-list file must raise a generic `Exception` wrapping the underlying `IOError`.
- A readable migrations-list file must return a list of `(app_label, migration_name)` tuples.
- An empty valid migrations-list file must return an empty list.
- The CLI must distinguish between no file (`None`), an empty file (`[]`), and an explicit set of migrations (list of tuples).

## Removed Noise

- Version number reference.
- Direct mention of a specific source line and GitHub URL.
- Implementation diagnosis about a method being called with every line instead of the filename.
- Repository/file path details that are not part of the public API contract.

## Risk Notes

- The original prompt uses `--include-migration-from`, while the surrounding text says `--include-migrations-from`; the rewritten task preserves the behavior without relying on the inconsistent flag spelling.
- The exact CLI wiring is not specified, so the agent must infer where the migration-list result is consumed while preserving the public reader contract.

## Original Prompt

Bug: --include-migrations-from argument being ignored
In version 2.2.2, using the `--include-migration-from` argument and specifying a migration .py file will not work and `lintmigrations` will run on all migration files.

On [line 299](https://github.com/3YOURMIND/django-migration-linter/blob/799957a5564e8ca1ea20d7cf643abbc21db4e40f/django_migration_linter/migration_linter.py#L299) of `migration_linter.py` the method `is_migration_file` is being called with every line of the `migrations_file_path` file instead of the filename `migrations_file_path`

## Original Interface

Method: MigrationLinter.read_migrations_list(cls, migrations_file_path)
Location: django_migration_linter/migration_linter.py
Inputs: migrations_file_path – a string path to a migrations‑list file or ``None``. ``None`` (or any falsy value) signals “no file specified”. A non‑existent or unreadable path triggers an IOError.
Outputs: • Returns ``None`` when ``migrations_file_path`` is falsy (no file was given). • Raises ``Exception`` (wrapping the original IOError) when the file cannot be opened. • Otherwise returns a ``list`` of ``(app_label, migration_name)`` tuples parsed from the file; an empty list indicates the file contained no valid lines.
Description: Class‑method that reads a plain‑text file listing migration identifiers. It distinguishes three situations required by the linter’s CLI: “all migrations” (``None`` → ``None``), “no migrations” (empty list), and an explicit set of migrations (list of tuples). Errors while accessing the file are escalated as a generic ``Exception``.
