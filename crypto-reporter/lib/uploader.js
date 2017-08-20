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

const Storage = require('@google-cloud/storage');
const config = require('../config');
const fs = require('fs');
const csvWriter = require('csv-write-stream')
const csvReader = require('csv-streamify')
const Promise = require('promise');

const CLOUD_BUCKET = config.get('CLOUD_BUCKET');

const storage = Storage({
  projectId: config.get('GCLOUD_PROJECT')
});

const bucket = storage.bucket(CLOUD_BUCKET);

function reportToCSV(coinName, price, high, low, volume, marketCap) {
  let fileName = `${coinName}.csv`;
  
  return  bucket.file(fileName).exists().then((data) => {
      let currentDate = _getCurrentDate()
      let exists = data[0];

      if (exists) {
        return _reportToExistingCSV(fileName, coinName, currentDate, price, high, low, volume, marketCap);
      } else {
        return _reportToNewCSV(fileName, currentDate, price, high, low, volume, marketCap);  
      }
    }
  );  
}

function _verifyNotReported(file, currDate) {
  return new Promise(function (resolve, reject) {  
    var remoteReadStream = file.createReadStream();
    const parser = csvReader()

    var dateNotReported = true;

    remoteReadStream.pipe(parser);

    parser.on('data', (line) => {
      let row = JSON.parse(line.toString())
      let date = row[0]
      let price = row[3]
      let volume = row[6]

      if (date == currDate) {
        console.log("Already reported date!: " + date);
        dateNotReported = false;
      }
    })

    parser.on('end', () => {
      resolve(dateNotReported);
    })
      
    parser.on('error', function(err) {
      reject(err);
    })  
  });
}

function _replaceFiles(newFileName, fileName) {
  return new Promise(function (resolve, reject) {  
    return bucket.file(fileName).delete()
      .then(() => {
        console.log(`gs://${CLOUD_BUCKET}/${fileName} deleted.`);
        
        bucket.file(newFileName).move(fileName).then(() => {
          console.log(`gs://${CLOUD_BUCKET}/${newFileName} moved to gs://${CLOUD_BUCKET}/${fileName}.`);
          resolve();
        })
        .catch((err) => {
          reject(err);
          console.error('ERROR:', err);  
        })
      })
      .catch((err) => {
        reject(err);
        console.error('ERROR:', err);
      });
  });
}

function _appendToCSV(remoteWriteStream, currentDate, price, high, low, volume, marketCap) {
    console.log(`Appending line... ${currentDate}, ${price}, ${high}, ${low} ${volume} ${marketCap}`)

    var writer = csvWriter({sendHeaders: false})
    writer.pipe(remoteWriteStream)
    writer.end({date: currentDate,
                dummy: 666666, 
                price: price,
                high: high,
                low: low,
                volume: volume,
                marketCap: marketCap});
}

function _getCurrentDate() {
  return new Date().toISOString().split('T')[0].replace(/-/g,'');
}

function _reportToNewCSV(fileName, currentDate, price, high, low, volume, marketCap) {
  var remoteWriteStream = bucket.file(fileName).createWriteStream();
  _appendToCSV(remoteWriteStream, currentDate, price, high, low, volume, marketCap);
}

function _reportToExistingCSV(fileName, coinName, currentDate, price, high, low, volume, marketCap) {

  let file = bucket.file(fileName);  
  let newFileName = `${coinName}_new.csv`;

  return new Promise(function (resolve, reject) {  
    _verifyNotReported(file, currentDate).then((newDate) => {  
      if (newDate) {
        var remoteReadStream = file.createReadStream();
        var newFile = bucket.file(newFileName);
        var remoteWriteStream = newFile.createWriteStream();
        remoteReadStream.pipe(remoteWriteStream);

        remoteReadStream.on('finish', () => {    
          _appendToCSV(remoteWriteStream, currentDate, price, high, low, volume, marketCap);

          remoteWriteStream.on('finish', () => {
            _replaceFiles(newFileName, fileName).then(() => {
              resolve();
            });
          })          
        });
      } else {
        reject('Date already reported!');
      }  
    }, (error) => {
      reject(error);
      console.log(`Error: ${error}`);
    });
  });
}

module.exports = {
  reportToCSV
};
