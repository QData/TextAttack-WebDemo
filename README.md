# TextAttack Web Demo

# Setup
Prior to starting, make sure you have > Python 3.6 installed on your machine with the latest version of pip

## First Time Setup and Installation
1. `git clone <repo_url>`
2. `cd TextAttack-WebDemo`
3. `python3 -m venv venv` to initialize the virtual environment
4. `source venv/bin/activate` [on MAC] or `source venv/Scripts/activate` [on WINDOWS] in order to activate the virtual environment
5. `pip install -r requirements.txt` to install the dependencies
   1. This process may fail on windows with `ModuleNotFoundError: No module named 'tools.nnwrap'`, in this case, run the following command and then rerun the pip install of requirements.txt: `pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html`

## Running the Web App
1. `python3 manage.py migrate` in order to initialize any database and models changes locally with sqlite
2. `python3 manage.py runserver` to create an instance of the server on your local machine
3. Visit `localhost:8000` in order to view the web application

## Debugging
If the `runserver` command fails, it is most likely a database issue. In this case, please refer to the following docs for help https://docs.djangoproject.com/en/3.1/topics/migrations/.