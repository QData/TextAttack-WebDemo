#!/bin/bash

source /var/app/venv/*/bin/activate
cd /var/app/staging

python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --noinput