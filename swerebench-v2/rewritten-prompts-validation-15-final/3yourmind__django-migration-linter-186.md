# 3yourmind__django-migration-linter-186

- repo: 3YOURMIND/django-migration-linter
- language: python
- difficulty: medium

## Rewritten Prompt

The linter should not warn about `CREATE INDEX` when the index is created in the same migration transaction as a new table. In a migration that creates a model and then emits non-concurrent `CREATE INDEX` statements for that model’s columns, those statements should be treated as safe and must not trigger the “CREATE INDEX locks table” warning.

This should work the same way as the existing `ADD_UNIQUE` handling: if the indexed table is created in the same transaction, return no warning; otherwise, keep warning on non-concurrent index creation. `CREATE INDEX CONCURRENTLY` should remain excluded from the lock warning.

The public analyser helper `has_create_index(sql_statements, **kwargs)` must keep its current contract: it takes the SQL statements for a single migration transaction and returns a boolean indicating whether a lock warning should be raised.

## Preserved Requirements

- Non-concurrent `CREATE INDEX` statements should still be detected as lock-causing unless the indexed table is also created in the same migration transaction.
- `CREATE INDEX` created in the same transaction as the table creation must not trigger the “CREATE INDEX locks table” warning.
- The behavior should mirror the existing `ADD_UNIQUE` logic for recognizing table creation in the same migration.
- `CREATE INDEX CONCURRENTLY` must not be treated as the lock-warning case.
- The public helper `has_create_index(sql_statements, **kwargs)` must remain available with the same input/output contract: iterable SQL statements for one migration transaction, returning `True` or `False`.

## Removed Noise

- Django model example used only to illustrate the generated SQL.
- Full `sqlmigrate` output block, since it only demonstrates the failure symptom.
- External GitHub URL and source-line reference.
- Implementation hint phrased as a suggestion rather than a behavioral requirement.
- Issue/PR-style explanatory boilerplate and metadata.

## Risk Notes

- The exact SQL table name extraction needs to handle the table referenced by the index when deciding whether it was created in the same transaction.
- The warning suppression should apply only when the table creation is present in the same migration transaction, not for unrelated prior or later migrations.
- The contract around `**kwargs` is that they are accepted but currently unused; avoid changing call compatibility.

## Quality Warnings

- missing_edge_literal:shipments_shipmentmetadataalert
- missing_edge_literal:metadata_id

## Original Prompt

Linter fails on CREATE INDEX when creating a new table
Here is an example `CreateModel` from Django:

```python
migrations.CreateModel(
    name='ShipmentMetadataAlert',
    fields=[
        ('deleted_at', models.DateTimeField(blank=True, db_index=True, null=True)),
        ('created_at', common.fields.CreatedField(default=django.utils.timezone.now, editable=False)),
        ('updated_at', common.fields.LastModifiedField(default=django.utils.timezone.now, editable=False)),
        ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False, verbose_name='ID')),
        ('message', models.TextField(blank=True, null=True)),
        ('level', models.CharField(blank=True, choices=[('HIGH', 'high'), ('MEDIUM', 'medium'), ('LOW', 'low')], max_length=16, null=True)),
        ('type', models.CharField(blank=True, choices=[('MOBILE_DEVICE_ALERT', 'MOBILE_DEVICE_ALERT'), ('NON_ACTIVE_CARRIER', 'NON_ACTIVE_CARRIER'), ('OTHER', 'OTHER')], max_length=32, null=True)),
        ('subtype', models.CharField(blank=True, choices=[('DRIVER_PERMISSIONS', 'DRIVER_PERMISSIONS'), ('DRIVER_LOCATION', 'DRIVER_LOCATION'), ('OTHER', 'OTHER')], max_length=32, null=True)),
        ('occurred_at', models.DateTimeField(null=True)),
        ('clear_alert_job_id', models.UUIDField(default=None, null=True)),
        ('metadata', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='alerts', to='shipments.ShipmentMetadata')),
    ],
    options={
        'abstract': False,
    }
)
```

Here are the SQL statements that this spits out in `sqlmigrate`:

```sql
BEGIN;
--
-- Create model ShipmentMetadataAlert
--
CREATE TABLE "shipments_shipmentmetadataalert" ("deleted_at" timestamp with time zone NULL, "created_at" timestamp with time zone NOT NULL, "updated_at" timestamp with time zone NOT NULL, "id" uuid NOT NULL PRIMARY KEY, "message" text NULL, "level" varchar(16) NULL, "type" varchar(32) NULL, "subtype" varchar(32) NULL, "occurred_at" timestamp with time zone NULL, "clear_alert_job_id" uuid NULL, "metadata_id" uuid NOT NULL);
ALTER TABLE "shipments_shipmentmetadataalert" ADD CONSTRAINT "shipments_shipmentme_metadata_id_f20850e8_fk_shipments" FOREIGN KEY ("metadata_id") REFERENCES "shipments_shipmentmetadata" ("id") DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX "shipments_shipmentmetadataalert_deleted_at_c9a93342" ON "shipments_shipmentmetadataalert" ("deleted_at");
CREATE INDEX "shipments_shipmentmetadataalert_metadata_id_f20850e8" ON "shipments_shipmentmetadataalert" ("metadata_id");
COMMIT;
```

This is an error from the linter as it outputs the error `CREATE INDEX locks table`. But the table is being created within the migration, it just needs to recognize that.

It seems like the `CREATE INDEX` detection should work the same way that the `ADD_UNIQUE` detection works where it detects that the create table is happening in the same migration:

https://github.com/3YOURMIND/django-migration-linter/blob/db71a9db23746f64d41d681f3fecb9b066c87338/django_migration_linter/sql_analyser/base.py#L26-L40

## Original Interface

Function: has_create_index(sql_statements, **kwargs)
Location: django_migration_linter/sql_analyser/postgresql.py
Inputs:
- sql_statements (Iterable[str]): list (or any iterable) of SQL statements that belong to a single migration transaction.
- **kwargs: additional keyword arguments passed by the analyser (currently unused).
Outputs: bool – True when a non‑concurrent CREATE INDEX statement is found that would lock a table, False otherwise (e.g., when the index is created CONCURRENTLY or the indexed table is also created within the same transaction).
Description: Scans the supplied statements for a CREATE INDEX (excluding CREATE INDEX CONCURRENTLY). If such an index is detected, it checks whether the table referenced by the index is also created in the same transaction; if the table is created, the function returns False (no warning), otherwise it returns True to trigger the “CREATE INDEX locks table” warning.
