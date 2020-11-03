from django.db import models

class AttackResult(models.Model):
    input_string = models.CharField(max_length=1000)
    attack_type = models.CharField(max_length=200)
    output_string = models.CharField(max_length=1000)