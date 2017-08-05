
var bittrex = require('node.bittrex.api');
const config = require('../config');
const request = require('request');
const uploader = require('./uploader');
const Promise = require('promise');

bittrex.options({
  'apikey' : config.get('BITTREX_API_KEY'),
  'apisecret' : config.get('BITTREX_API_SECRET'), 
  'stream' : true, // will be removed from future versions 
  'verbose' : true,
  'cleartext' : true
  // 'baseUrlv2' : 'https://bittrex.com/Api/v2'
});

const MIN_MARKET_CAP = 100000000;

function reportStats() {

  var marketCaps = {};

  return new Promise(function (resolve, reject) {  
    request("https://api.coinmarketcap.com/v1/ticker/", (error, response, body) => {

      if (error) {
        let msg = `Error getting marketcaps: ${error}`;
        console.log(msg);
        reject(msg);
        return;
      }

      console.log('get market cap statusCode:', response && response.statusCode);
      let markets = JSON.parse(body);
      markets.forEach((c) => {
        marketCaps[c.symbol] = c.market_cap_usd ? parseFloat(c.market_cap_usd) : null;
      });

      bittrex.getmarketsummaries((data, err) => {        
        if (err) {
          console.error(err);
          reject(err)
          return;
        }

        if (Array.isArray(data)) {
          let btcUsd = data.find((m) => m["MarketName"] == "USDT-BTC")["Last"];
          let btcMarkets = data.filter((c) => c["MarketName"].indexOf("BTC-") != -1);

          promises = [];

          btcMarkets.forEach((market) => {

            let usdPrice = btcUsd * market["Last"];
            let usdHigh = btcUsd * market["High"];
            let usdLow = btcUsd * market["Low"];
            let usdVolume = btcUsd * market["Volume"];
            let coinName = market["MarketName"].split('-')[1];
            let marketCap = marketCaps[coinName];
          
            promise = uploader.reportToCSV(coinName, usdPrice, usdHigh, usdLow, usdVolume, marketCap);
            promises.push(promise);
          });

          return Promise.all(promises).then(() => {
            console.log("Finished uploading files... Notifying predictor");
            // TODO: Notify predictor
            resolve();
          }, (error) => {
            reject(error);
          });
        }
      });
    });
  });
}

module.exports = {
  reportStats
}
