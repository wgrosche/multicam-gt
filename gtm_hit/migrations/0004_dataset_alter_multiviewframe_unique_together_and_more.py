# Generated by Django 4.1.7 on 2023-04-06 06:40

from django.db import migrations, models
import django.db.models.deletion

def create_dataset(apps, schema_editor):
    Dataset = apps.get_model('gtm_hit', 'Dataset')
    new_dataset = Dataset(name='scout', description='SCOUT1 dataset')
    new_dataset.save()

class Migration(migrations.Migration):

    dependencies = [
        ("gtm_hit", "0003_alter_person_unique_together"),
    ]

    operations = [
        migrations.CreateModel(
            name="Dataset",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100, unique=True)),
                ("description", models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.RunPython(create_dataset),
        migrations.AlterUniqueTogether(
            name="multiviewframe",
            unique_together=set(),
        ),
        migrations.AddField(
            model_name="multiviewframe",
            name="dataset",
            field=models.ForeignKey(
                default=1,
                on_delete=django.db.models.deletion.CASCADE,
                to="gtm_hit.dataset",
            ),
        ),
        migrations.AddField(
            model_name="singleviewframe",
            name="dataset",
            field=models.ForeignKey(
                default=1,
                on_delete=django.db.models.deletion.CASCADE,
                to="gtm_hit.dataset",
            ),
        ),
        migrations.AlterUniqueTogether(
            name="multiviewframe",
            unique_together={("frame_id", "undistorted", "worker", "dataset")},
        ),
    ]
