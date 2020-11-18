#!/bin/bash

echo "Entering predeploy hook"
source /var/app/venv/*/bin/activate
cd /var/app/current
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --noinput