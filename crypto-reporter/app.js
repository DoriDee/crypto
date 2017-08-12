// Copyright 2017, Google, Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

const path = require('path');
const express = require('express');
const config = require('./config');

const app = express();

app.disable('etag');
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');
app.set('trust proxy', true);

const api = require('./lib/bittrex_api_runner.js')
const uploader = require('./lib/uploader.js')
const PubSub = require(`@google-cloud/pubsub`);

// api.reportStats();
function publishMessage () {
  // Instantiates a client
  const pubsub = PubSub();

  // References an existing topic, e.g. "my-topic"
  const topic = pubsub.topic(config.get('TOPIC_NAME'));

  // Publishes the message, e.g. "Hello, world!" or { amount: 599.00, status: 'pending' }
  return topic.publish('pickle rick!!')
    .then((results) => {
      const messageIds = results[0];
      console.log(`Message ${messageIds[0]} published.`);
      return messageIds;
    });
}

app.get('/publish', (req, res, next) => {
  console.log("COOCOO!!!")

  publishMessage();

  res.status(200).send('PICKLE RICKKKKK');
});

app.use('/report', (req, res, next) => {
  api.reportStats().then(() => {
    res.status(200).send('OK');
  }, (error) => {
    res.status(422).send(error);
  }); 
});


// Basic 404 handler
app.use((req, res) => {
  res.status(404).send('Not Found');
});

// Basic error handler
app.use((err, req, res, next) => {
  /* jshint unused:false */
  console.error(err);
  // If our routes specified a specific response, then send that. Otherwise,
  // send a generic message so as not to leak anything.
  res.status(500).send(err.response || 'Something broke!');
});

if (module === require.main) {
  // Start the server
  const server = app.listen(config.get('PORT'), () => {
    const port = server.address().port;
    console.log(`App listening on port ${port}`);
  });
}

module.exports = app;
