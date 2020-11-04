from django.db import models
from django.contrib.postgres.fields import ArrayField

from django.utils import timezone

class AttackResult(models.Model):
    # metadata
    session_id = models.CharField(max_length=100)
    timestamp = models.DateTimeField(default=timezone.now)

    # inputs
    input_string = models.CharField(max_length=1000)
    input_histogram = models.CharField(max_length=1000, default="[]")
    input_label = models.CharField(max_length=5)
    model_name = models.CharField(max_length=200)
    recipe_name = models.CharField(max_length=200)

    # outputs
    output_string = models.CharField(max_length=1000)
    output_label = models.CharField(max_length=5)
    output_histogram = models.CharField(max_length=1000, default="[]")