"use client"
import React, { useState, useEffect, useRef } from 'react';
import ChartComponent from '../components/ChartComponent';

const apiUrl = 'http://localhost:8000';

export default function ChartPage() {
  const [algorithm, setAlgorithm] = useState('long-term');
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
    setPredictData(null); // Reset predictData to null before fetching new data
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
        mape: data.mape || 0,
        rmse: data.rmse || 0,
        mae: data.mae || 0,
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

      const uniqueDates = new Set(); // Set to store unique dates
      const formattedData = data.data.reduce((accumulator, element) => {
        const date = new Date(element.date).toISOString().substring(0, 10);
        if (!uniqueDates.has(date)) {
          uniqueDates.add(date); // Add unique date to the Set
          accumulator.push({
            time: date,
            open: element.open || 0,
            close: element.close || 0,
            high: element.high || 0,
            low: element.low || 0,
            volume: element.volume || 0,
            change: element.change || 0,
          });
        }
        return accumulator;
      }, []).sort((a, b) => a.time.localeCompare(b.time)); // Sort the array by time (date)

      setData(formattedData);
      historyDataRef.current = formattedData;
      setIsLoading(false); // Set loading state to false after data has been fetched
    } catch (error) {
      console.error(error);
      setIsLoading(false); // Set loading state to false in case of an error
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between md:p-24">
      <div className="z-10 w-full md:max-w-8xl max-h-[80%] items-center justify-between font-mono text-sm lg:flex">
        {isLoading ? (
          <p>Loading...</p>
        ) : (
          <ChartComponent className="w-full h-full" data={data} predictData={predictData?.data || []} />
        )}
      </div>

      <div className="flex items-center justify-center mt-4">
        {['lstnet', 'lstm', 'xgboost', 'mtgnn', 'random_forest', 'var', 'long-term'].map((option) => (
          <label key={option} className="mr-4">
            <input
              type="radio"
              value={option}
              checked={algorithm === option}
              onChange={handleOptionChange}
            />
            {option}
          </label>
        ))}
      </div>

      <div className="flex items-center justify-center mt-4">
        <input
          type="range"
          min="1"
          max="100"
          value={predictLength}
          onChange={handleInputChange}
          className="rounded-md w-64"
        />
        <span className="ml-2">{predictLength} days</span>
        <button onClick={handleButtonClick} className="ml-4 px-4 py-2 bg-blue-500 text-white rounded-md">
          Execute API
        </button>
      </div>

      {predictData?.predictionMetric && (
        <div className="mt-4 border border-gray-300 rounded-md p-4">
          <h2 className="mb-2">Prediction Metric</h2>
          <p>
            <strong>MAPE:</strong> {predictData.predictionMetric.mape}
          </p>
          <p>
            <strong>RMSE:</strong> {predictData.predictionMetric.rmse}
          </p>
          <p>
            <strong>MAE:</strong> {predictData.predictionMetric.mae}
          </p>
        </div>
      )}
    </main>

  );

}
