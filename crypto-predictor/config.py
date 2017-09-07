
import os

# Google Cloud Project ID. This can be found on the 'Overview' page at
# https://console.developers.google.com
PROJECT_ID = 'cryptopred'
TOPIC_NAME = 'predictions'
DATA_BACKEND = 'cloudsql'

CLOUDSQL_USER = 'postgres'
CLOUDSQL_PASSWORD = '5KNIhq26yhPny2pf'
CLOUDSQL_DATABASE = 'cryptodb'
CLOUDSQL_CONNECTION_NAME = 'cryptopred:us-central1:cryptodb'

SUBSCRIPTION_NAME = 'predictor'

LOCAL_SQLALCHEMY_DATABASE_URI = 'postgresql://localhost/crypto'    
    # 'mysql+pymysql://{user}:{password}@127.0.0.1:3306/{database}').format(
    #     user=CLOUDSQL_USER, password=CLOUDSQL_PASSWORD,
    #     database=CLOUDSQL_DATABASE)
DB_IP = "104.155.151.224"

# When running on App Engine a unix socket is used to connect to the cloudsql
# instance.
# LIVE_SQLALCHEMY_DATABASE_URI = (
#     'postgresql://{user}:{password}@localhost/{database}'
#     '?unix_socket=/cloudsql/{connection_name}').format(
#         user=CLOUDSQL_USER, password=CLOUDSQL_PASSWORD,
#         database=CLOUDSQL_DATABASE, connection_name=CLOUDSQL_CONNECTION_NAME)


LIVE_SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:5KNIhq26yhPny2pf@104.155.151.224/postgres'

# LIVE_SQLALCHEMY_DATABASE_URI = (
#     'postgresql://{user}:{password}@{ip}/postgres').format(
#         user=CLOUDSQL_USER, password=CLOUDSQL_PASSWORD, ip=DB_IP,
#         database=CLOUDSQL_DATABASE, connection_name=CLOUDSQL_CONNECTION_NAME)

# if os.environ.get('GAE_INSTANCE'):
SQLALCHEMY_DATABASE_URI = LIVE_SQLALCHEMY_DATABASE_URI
# else:
#     SQLALCHEMY_DATABASE_URI = LOCAL_SQLALCHEMY_DATABASE_URI
