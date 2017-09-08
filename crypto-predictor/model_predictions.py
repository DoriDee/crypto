# Copyright 2015 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import time, datetime
from datetime import timedelta, date

builtin_list = list

db = SQLAlchemy()

def init_app(app):
    # Disable track modifications, as it unnecessarily uses memory.
    app.config.setdefault('SQLALCHEMY_TRACK_MODIFICATIONS', False)
    db.app = app
    db.init_app(app)


def from_sql(row):
    """Translates a SQLAlchemy model instance into a dictionary"""
    data = row.__dict__.copy()
    data['id'] = row.id
    data.pop('_sa_instance_state')
    return data


# [START model]
class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    coin_symbol = db.Column(db.String(255))
    last_value = db.Column(db.Float)
    prediction = db.Column(db.Float)
    real_value = db.Column(db.Float)
    market_cap = db.Column(db.Float)
    predicted_at = db.Column(db.Date)
    
    def __repr__(self):
        return "<Prediction(coin_symbol='%f', last_value=%f, prediction=%f)" % (self.coin_symbol, self.last_value, self.prediction)
# [END model]


# [START list]
def list(date, limit=10, cursor=None):
    cursor = int(cursor) if cursor else 0
    query = (Prediction.query
             .filter(Prediction.predicted_at==date)
             .order_by(Prediction.market_cap)
             .limit(limit)
             .offset(cursor))
    coins = builtin_list(map(from_sql, query.all()))
    next_page = cursor + limit if len(coins) == limit else None
    return (coins, next_page)
# [END list]


def get_last_prediction(coin_symbol):
    print("Date: {0} Symbol: {1}".format(date.today() - timedelta(1), coin_symbol))

    result = Prediction.query.filter(Prediction.predicted_at == date.today() - timedelta(1), 
                                    Prediction.coin_symbol == coin_symbol).first()

    if not result:
        return None
    return from_sql(result)

# [START read]
def read(id):
    result = Prediction.query.get(id)
    if not result:
        return None
    return from_sql(result)
# [END read]

def is_exists(coin_symbol):
    result = Prediction.query.filter(Prediction.predicted_at == date.today(), 
                                     Prediction.coin_symbol == coin_symbol).first()

    return result

# [START create]
def create(data):
    if (not is_exists(data['coin_symbol'])):
        prediction = Prediction(**data)
        db.session.add(prediction)
        db.session.commit()
        return from_sql(prediction)
    else:
        print("Coin: {0} already exists for date: {1}".format(data['coin_symbol'], date.today()))
        return None

    
# [END create]


# [START update]
def update(data, id):
    pred = Prediction.query.get(id)
    for k, v in data.items():
        setattr(pred, k, v)
    db.session.commit()
    return from_sql(pred)
# [END update]


def delete(id):
    Prediction.query.filter_by(id=id).delete()
    db.session.commit()


def connect():
    app = Flask(__name__)
    app.config.from_pyfile('./config.py')

    with app.app_context():
        init_app(app)        
        
    print("Connected..")

def _create_database():
    """
    If this script is run directly, create all the tables necessary to run the
    application.
    """
    app = Flask(__name__)
    app.config.from_pyfile('./config.py')
    init_app(app)
    with app.app_context():
        db.create_all()
    print("All tables created")


if __name__ == '__main__':
    _create_database()
