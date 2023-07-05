"use client"
import React, { useState, useEffect, useRef } from 'react';
import ChartComponent from '../components/ChartComponent';
import Navbar from '../components/Navbar';

const apiUrl = 'https://vn30-api.ftisu.vn';

export default function ChartPage() {
  const [algorithm, setAlgorithm] = useState('long-term');
  const [algorithmMetric, setAlgorithmMetric] = useState('');
  const [predictLength, setPredictLength] = useState(7);
  const [data, setData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [predictData, setPredictData] = useState(null);
  const historyDataRef = useRef([]);

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (predictData) {
      setData([...historyDataRef.current, ...predictData.data]);
    }
  }, [predictData]);

  const handleOptionChange = (event) => {
    setAlgorithm(event.target.value);
  };

  const handleInputChange = (event) => {
    setPredictLength(event.target.value);
  };

  const handleButtonClick = () => {
    setPredictData(null);
    setAlgorithmMetric(algorithm) // Reset predictData to null before fetching new data
    fetchPredictData();
  };

  async function fetchPredictData() {
    const endpoint = algorithm === 'long-term' ? `vn30/long-term` : algorithm;
    const url = `${apiUrl}/predict/${endpoint}?pred_len=${predictLength}`;

    try {
      const response = await fetch(url);
      const data = await response.json();

      const formattedData = data.data.VN30.map(({ date, close, open, high, low, volume, change }) => ({
        open: open || 0,
        close: close || 0,
        high: high || 0,
        low: low || 0,
        volume: volume || 0,
        time: date.substring(0, 10),
        change: change || 0,
      })).sort((a, b) => a.time.localeCompare(b.time));

      const predictionMetric = {
        ape: data.predictionMetric?.ape || 0,
        me: data.predictionMetric?.me || 0,
        mae: data.predictionMetric?.mae || 0,
        mpe: data.predictionMetric?.mpe || 0,
        rmse: data.predictionMetric?.rmse || 0,
        corr: data.predictionMetric?.corr || 0,
        minmax: data.predictionMetric?.minmax || 0,
      };

      const formattedDataWithMetric = {
        data: formattedData,
        predictionMetric: predictionMetric,
      };
      setPredictData(formattedDataWithMetric);
    } catch (error) {
      console.error(error);
    }
  }

  async function fetchData() {
    try {
      const res = await fetch(`${apiUrl}/history`);
      const data = await res.json();

      // const uniqueDates = new Set(); // Set to store unique dates
      const formattedData = data.data.reduce((accumulator, element) => {
        accumulator.push({
          time: element.date || "2023-05-25T00:00:00",
          open: element.open || 0,
          close: element.close || 0,
          high: element.high || 0,
          low: element.low || 0,
          volume: element.volume || 0,
          change: element.change || 0,
        });
        return accumulator;
      }, []).sort((a, b) => a.time.localeCompare(b.time)); // Sort the array by time (date)
      console.log("formated data:", formattedData);
      setData(formattedData);
      historyDataRef.current = formattedData;
      setIsLoading(false); // Set loading state to false after data has been fetched
    } catch (error) {
      console.error(error);
      setIsLoading(false); // Set loading state to false in case of an error
    }
  }
  return (
    <div>
      <Navbar />
      <main className="flex min-h-screen bg-white">
        <div className="w-10/12 p-10">
          {isLoading ? (
            <p>Loading...</p>
          ) : (
            <ChartComponent className="w-full h-full" data={data} predictData={predictData?.data || []} />
          )}
        </div>

        <div className="w-2/12 pt-20 px-10">
          <div className="bg-blue-300 rounded-md p-4 text-black p-10">
            <div className="flex flex-col items-start justify-center">
              <div className="mb-4">
                <label htmlFor="algorithm" className="mb-2 text-lg font-bold">
                  Algorithm:
                </label>
                <select id="algorithm" value={algorithm} onChange={handleOptionChange} className="border border-gray-300 rounded-md px-2 py-1 w-full">
                  <option value="lstm">LSTM</option>
                  <option value="xgboost">XGBOOST</option>
                  <option value="random_forest">Random Forest</option>
                  <option value="var">VAR</option>
                  <option value="mtgnn">MTGNN</option>
                  <option value="lstnet">LSTNET</option>
                  <option value="long-term">Long Term</option>
                </select>
              </div>

              <div className="mb-4">
                <label htmlFor="predictLength" className="mb-2 text-lg font-bold ">
                  Prediction Length:
                </label>
                <select id="predictLength" value={predictLength} onChange={handleInputChange} className="border border-gray-300 rounded-md px-2 py-1 w-full">
                  <option value="1">1 day</option>
                  <option value="7">7 days</option>
                  <option value="30">1 month </option>
                  <option value="90">3 months</option>
                </select>
              </div>

              <button onClick={handleButtonClick} className="w-full px-4 py-2 bg-blue-500 text-white rounded-md font-bold">
                Predict
              </button>
              {predictData?.predictionMetric && (
                <div className="mt-4 rounded-lg bg-blue-100 p-3 w-full">
                  <p className="text-lg font-bold mb-2">{algorithmMetric === '' ? 'Algorithm' : algorithmMetric}</p>
                  <div className="flex flex-col">
                    <p className="text-gray-900">MAPE: {predictData.predictionMetric.mpe}%</p>
                    <p className="text-gray-900">MAE: {predictData.predictionMetric.mae}</p>
                    <p className="text-gray-900">RMSE: {predictData.predictionMetric.rmse}</p>
                    <p className="text-gray-900">Corr: {predictData.predictionMetric.corr}%</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );

}