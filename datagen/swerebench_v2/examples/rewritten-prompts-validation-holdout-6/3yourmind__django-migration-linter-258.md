# 3yourmind__django-migration-linter-258

- repo: 3YOURMIND/django-migration-linter
- language: python
- difficulty: easy

## Rewritten Prompt

A migration that creates an index with a partial `WHERE` clause containing `IS NOT NULL` should not be reported as a `NOT NULL constraint on columns` error. For example, creating a concurrent index with a condition like `data_deleted_at IS NULL AND delete_data_after IS NOT NULL` is a safe index operation and must be allowed by the linter.

At the moment, this kind of `CREATE INDEX CONCURRENTLY ... WHERE (...)` SQL is incorrectly flagged just because the generated statement contains `NOT NULL`. The linter should distinguish between real `NOT NULL` constraints and index definitions that happen to include `IS NOT NULL` in their predicate.

## Preserved Requirements

- A partial index created with a `WHERE` clause containing `IS NOT NULL` must not be flagged as a `NOT NULL constraint on columns` error.
- `CREATE INDEX CONCURRENTLY` statements with such predicates should be treated as safe operations.
- The observable behavior to preserve is that `NOT NULL` inside an index predicate is not the same as a column `NOT NULL` constraint.

## Removed Noise

- Step-by-step reproduction section.
- Specific migration snippet and generated SQL example.
- Internal implementation diagnosis and suggested regex change.
- Source link to repository code.
- Issue/PR template boilerplate and metadata.

## Risk Notes

- The exact boundary between safe index predicates and true `NOT NULL` constraints should remain intact.
- Behavior should still flag genuine `NOT NULL` constraints outside of index definitions.

## Original Prompt

Adding an index with a NOT NULL condition incorrectly triggers NOT_NULL rule
Adding an index with a `WHERE` clause including `NOT NULL` gets flagged as a `NOT NULL constraint on columns` error.

## Steps to reproduce

The follow migration operation:

```python
AddIndexConcurrently(
    model_name="prediction",
    index=models.Index(
        condition=models.Q(
            ("data_deleted_at__isnull", True),
            ("delete_data_after__isnull", False),
        ),
        fields=["delete_data_after"],
        name="delete_data_after_idx",
    ),
),
```

Generates the following SQL:

```sql
CREATE INDEX CONCURRENTLY "delete_data_after_idx" ON "models_prediction" ("delete_data_after") WHERE ("data_deleted_at" IS NULL AND "delete_data_after" IS NOT NULL);
```

When linted this is flagged as an error because of the `NOT NULL`, when it ought to be a safe operation.

## Investigation

Looking at the condition used for this rule, I think it might just need to permit `CREATE INDEX` requests:

```python
re.search("(?<!DROP )NOT NULL", sql) and not sql.startswith("CREATE TABLE") and not sql.startswith("CREATE INDEX")
```

https://github.com/3YOURMIND/django-migration-linter/blob/202a6d9d5dea83528cb52fd7481a5a0565cc6f83/django_migration_linter/sql_analyser/base.py#L43

## Original Interface

No new interfaces are introduced.
