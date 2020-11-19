#!/bin/bash

echo "Entering predeploy hook"
source /var/app/venv/*/bin/activate

python manage.py collectstatic --noinput