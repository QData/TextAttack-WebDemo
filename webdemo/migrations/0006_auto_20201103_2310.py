# Generated by Django 3.1.3 on 2020-11-04 04:10

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('webdemo', '0005_auto_20201103_2300'),
    ]

    operations = [
        migrations.AddField(
            model_name='attackresult',
            name='session_id',
            field=models.CharField(default=None, max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='attackresult',
            name='timestamp',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]
