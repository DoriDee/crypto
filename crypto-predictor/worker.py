"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division

import base64
import json
import time, datetime
import requests
import click

from google.cloud import pubsub
from google.cloud import storage
from google.cloud import error_reporting

from logger import Logger
from recurror import Recurror

import predictor
import model_predictions

METADATA_URL_PROJECT = "http://metadata/computeMetadata/v1/project/"
METADATA_URL_INSTANCE = "http://metadata/computeMetadata/v1/instance/"
METADTA_FLAVOR = {'Metadata-Flavor' : 'Google'}

# Get the metadata related to the instance using the metadata server
PROJECT_ID = requests.get(METADATA_URL_PROJECT + 'project-id', headers=METADTA_FLAVOR).text
INSTANCE_ID = requests.get(METADATA_URL_INSTANCE + 'id', headers=METADTA_FLAVOR).text
INSTANCE_NAME = requests.get(METADATA_URL_INSTANCE + 'hostname', headers=METADTA_FLAVOR).text
INSTANCE_ZONE_URL = requests.get(METADATA_URL_INSTANCE + 'zone', headers=METADTA_FLAVOR).text
INSTANCE_ZONE = INSTANCE_ZONE_URL.split('/')[0]
TOPIC_NAME = 'predictions'
SUBSCRIPTION_NAME = 'predictions-sub'
BUCKET_NAME = 'coins-history'

REFRESH_INTERVAL = 25

def main():
    """
    """

    client = error_reporting.Client()

    data = {'coin_symbol': 'LTC', 'last_value': 555, 'prediction': 1231, 'real_value': 123, 'market_cap': 11919, 'predicted_at': datetime.datetime.now()}

    model_predictions.connect()
    prediction = model_predictions.create(data)

    return
    topic = pubsub_client.topic(TOPIC_NAME)
    subscription = topic.subscription(SUBSCRIPTION_NAME)

    Logger.log_writer("Main entry!!!")

    if not subscription.exists():
        sys.stderr.write('Cannot find subscription {0}\n'.format(sys.argv[1]))
        return

    Logger.log_writer("Wallak exists akakakak!!!")

    r = Recurror(REFRESH_INTERVAL - 10, postpone_ack)

    # pull() blocks until a message is received
    while True:
        try:            
            resp = subscription.pull(return_immediately=False)
            
            Logger.log_writer("pulled it: {0}".format(resp))

            for ack_id, message in resp:

                Logger.log_writer("ack_id:{0} message:{1}".format(ack_id, message))

                # We need to do this to get contentType. The rest is in attributes
                #[START msg_format]
                data = message.data

                Logger.log_writer("msg_data: {0}".format(data))
                #[END msg_format]

                # Start refreshing the acknowledge deadline.
                r.start(ack_ids=[ack_id], refresh=REFRESH_INTERVAL, sub=subscription)

                start_process = datetime.datetime.now()

                Logger.log_writer("Predicting....")

                parse_files()
        
                end_process = datetime.datetime.now()

                #[START ack_msg]
                # Delete the message in the queue by acknowledging it.
                subscription.acknowledge([ack_id])
                #[END ack_msg]

                # Write logs only if needed for analytics or debugging
                Logger.log_writer(
                    "processed by instance {instance_hostname} in {amount_time}"
                    .format(
                        instance_hostname=INSTANCE_NAME,
                        amount_time=str(end_process - start_process)
                    )
                )

                # Stop the ackDeadLine refresh until next message.
                r.stop()
        except Exception:
            client.report_exception()


def parse_files():
    bucket = gcs_client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()

    Logger.log_writer("Reading bucket files....")

    for blob in blobs:

        Logger.log_writer("fileName:" + blob.name)

        if blob.name == "LTC.csv":

            Logger.log_writer("Downloading...")

            blob.download_to_filename(blob.name)

            Logger.log_writer("Downloaded! Predicting...")

            last_value, prediction, market_cap = predictor.predict(blob.name)

            Logger.log_writer("Created a prediction for:{0} last_value:{1} prediction:{2} marketcap:{3}".format(blob.name,last_value,prediction,market_cap))


# https://github.com/GoogleCloudPlatform/pubsub-media-processing/blob/master/worker.py
def parse_bucket_object(params):
    # msg_data: { "kind": "storage#object", "id": "coins-history/README.md/1502556022304897", 
    # "selfLink": "https://www.googleapis.com/storage/v1/b/coins-history/o/README.md", 
    # "name": "README.md", "bucket": "coins-history", "generation": "1502556022304897", 
    # "metageneration": "1", "contentType": "application/octet-stream", 
    # "timeCreated": "2017-08-12T16:40:22.283Z", "updated": "2017-08-12T16:40:22.283Z", 
    # "storageClass": "REGIONAL", "timeStorageClassUpdated": "2017-08-12T16:40:22.283Z",
    #  "size": "9", "md5Hash": "O2QO+au3DPjNmLNCSWkTHg==", 
    # "mediaLink": "https://www.googleapis.com/download/storage/v1/b/coins-history/o/README.md?generation=1502556022304897&alt=media",
    #  "crc32c": "JdSPcQ==", "etag": "CIHh26+R0tUCEAE=" }
    msg_data = json.loads(params)
    content_type = msg_data["contentType"]
    # <Your custom process>
    if event_type == 'OBJECT_FINALIZE':
        Logger.log_writer("PICKKKEEE RICKKKKKK MADAFUCKAAAA!!! should predict right heree biaaatch")                    
    # <End of your custom process>


def postpone_ack(params):
    """Postpone the acknowledge deadline until the media is processed
    Will be paused once a message is processed until a new one arrives
    Args:
        ack_ids: List of the message ids in the queue
    Returns:
        None
    Raises:
        None
    """
    ack_ids = params['ack_ids']
    refresh = params['refresh']
    sub = params['sub']
    Logger.log_writer(','.join(ack_ids) + ' postponed')

    #[START postpone_ack]
    #Increment the ackDeadLine to make sure that file has time to be processed
    sub.modify_ack_deadline(ack_ids, refresh)    # API request
    #[END postpone_ack]

"""Create the API clients."""
pubsub_client = pubsub.Client(PROJECT_ID)
gcs_client = storage.Client(PROJECT_ID)

"""Launch the loop to pull media to process."""
if __name__ == '__main__':
    main()
