# 3yourmind__django-migration-linter-222

- repo: 3YOURMIND/django-migration-linter
- language: python
- difficulty: easy

## Rewritten Prompt

The linter should not incorrectly fail data migrations that use a ManyToMany field’s `through` model via an `apps.get_model(...)` lookup on the parent model. When a migration function accesses `Question.many_to_may.through.objects.bulk_create(...)`, it should be treated as valid usage and should not trigger the error saying it could not find an `apps.get_model("...", "through")` call.

Keep the existing rule that importing models directly in data migrations is incorrect, but make sure this `through`-model access pattern is recognized as safe. The reported behavior should no longer flag this case as an invalid direct model import.

## Preserved Requirements

- Data migrations must still reject direct model imports.
- Accessing a ManyToMany field’s `through` model through `apps.get_model(...)` on the related model should be accepted.
- The specific `Question.many_to_may.through.objects.bulk_create(...)` pattern should not raise the "Could not find an 'apps.get_model(\"...\", \"through\")' call" error.

## Removed Noise

- Documentation URL to Django’s `through` docs.
- Issue-style example code block formatting and commentary.
- The example migration error block as a quoted stack/error excerpt.
- Repository metadata and benchmark framing.
- Placeholder text and line-level diagnosis hint.

## Risk Notes

- The field name `many_to_may` may be a typo in the original example; preserve the behavior for the shown pattern without assuming a renamed API.
- The linter’s broader detection rules for valid `through` model usage should remain intact while eliminating this false positive.

## Original Prompt

Linter failing when using django 'through'
### through doc
https://docs.djangoproject.com/en/4.0/ref/models/fields/#django.db.models.ManyToManyField.through

### Example code
```
def forwards_func(apps, schema_editor):
    Question = apps.get_model("solution", "Question")
    ...
    Question.many_to_may.through.objects.bulk_create(...)                 <- this line?
    ...
```
### Example Error
```
(fs_solution, 0002_my_migration)... ERR (cached)
        'forwards_func': Could not find an 'apps.get_model("...", "through")' call. Importing the model directly is incorrect for data migrations.
```

## Original Interface

No new interfaces are introduced.
