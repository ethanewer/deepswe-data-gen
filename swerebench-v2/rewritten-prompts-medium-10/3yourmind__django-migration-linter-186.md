# 3yourmind__django-migration-linter-186

- repo: 3YOURMIND/django-migration-linter
- language: python
- difficulty: medium

## Rewritten Prompt

The linter should not warn about `CREATE INDEX` when the index is created as part of creating a new table in the same migration transaction. In that case, the table does not already exist, so the index creation should be treated as safe and should not trigger the “CREATE INDEX locks table” warning.

Keep the existing behavior for ordinary non-concurrent `CREATE INDEX` statements on existing tables. Also keep ignoring `CREATE INDEX CONCURRENTLY`.

If a migration transaction includes both the table creation and the index creation for that same table, suppress the warning for that index.

## Preserved Requirements

- Do not warn about `CREATE INDEX` when the indexed table is created in the same migration transaction.
- Continue warning for non-concurrent `CREATE INDEX` on existing tables.
- Do not warn for `CREATE INDEX CONCURRENTLY`.
- Treat a table created in the same transaction as not causing a lock-warning for its index creation.

## Removed Noise

- Issue title and example Django migration snippet.
- Example SQL output from sqlmigrate.
- Reference to the linter error text as an example.
- External GitHub URL and line-number reference.
- Mention of ADD_UNIQUE as an implementation comparison hint.
- Generated interface notes and explicit function signature metadata.
- Repository/package metadata and language label.
- References to tests, patches, PRs, and hidden implementation guidance.

## Risk Notes

- The prompt does not specify how the analyzer determines which table a CREATE INDEX targets; the implementation must infer that from the SQL.
- If multiple tables are created in one migration, the behavior should apply only when the indexed table is among them.
- The warning should remain based on non-concurrent CREATE INDEX statements only.

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
