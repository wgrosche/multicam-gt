

# EPFL Multi-camera detection annotation tool
Our web-application to annotate multi-camera detection datasets. Feel free to download and use it to annotate other datasets.

#

## Usage
Define the DATABASES variable in the settings.py file to point to your database. The application is configured to use a PostgreSQL database. You can use the following configuration:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'invision',
        'USER': 'invision',
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': '',
    }
}
```
You can load the INVISION data into the database by running the following command:
```bash
python manage.py shell < invision_data_process.py
```
You can then run the following commands to run the application:
```bash
conda env create -n ENVNAME --file ENV.yml
pip install -r requirements.txt
initdb -D invision   
chmod -R 700 invision
pg_ctl -D invision -l logfile start
createuser invision
createdb --owner=invision invision

python manage.py migrate
python manage.py runserver 0.0.0.0:4444
```
You can now access the application at http://localhost:4444



### Acknowledgements
Written by Stephane Bouquet during his MSc. Thesis at CVLab, EPFL, under the supervision of Pierre Baqué.
We acknowledge the WILDTRACK project and the Swiss National Scientific Fund.
