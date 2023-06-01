"use client"
import React, { useState, useEffect } from 'react';
import ChartComponent from '../components/ChartComponent';


export default function ChartPage() {
  const [algorithm, setAlgorithm] = useState('');
  const [predictLength, setPredictLength] = useState(7);
  const [data, setData] = useState();
  const [isLoading, setIsLoading] = useState(true);
  const [predictData, setPredictData] = useState();

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (predictData) {
      setData((prevData) => [...prevData.slice(0, -predictData.length), ...predictData]);
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
    const apiUrl = algorithm === "long-term"
      ? `http://localhost:8000/predict/vn30/long-term?pred_len=${predictLength}`
      : `http://localhost:8000/predict/${algorithm}?pred_len=${predictLength}`;

    try {
      const response = await fetch(apiUrl);
      const data = await response.json();

      const formattedData =
        data.data.VN30.map(({ date, close, open, high, low, volume, change }) => ({
          open: open || 0,
          close: close || 0,
          high: high || 0,
          low: low || 0,
          volume: volume || 0,
          time: date.substring(0, 10),
          change: change || 0,
        })).sort((a, b) => a.time.localeCompare(b.time));

      setPredictData(formattedData);
    } catch (error) {
      console.error(error);
    }
  }


  async function fetchData() {
    try {
      const res = await fetch(`http://localhost:8000/history`);
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
      console.log("formatedData :", formattedData)
      setData(formattedData);
      setIsLoading(false); // Set loading state to false after data has been fetched
    } catch (error) {
      console.error(error);
      setIsLoading(false); // Set loading state to false in case of an error
    }
  }


  return (
    <main className="flex min-h-screen flex-col items-center justify-between md:p-24">
      {/* Chart Area */}
      <div className="z-10 w-full md:max-w-8xl max-h-[80%] items-center justify-between font-mono text-sm lg:flex">
        {isLoading ? (
          <p>Loading...</p> // Show a loading indicator while data is being fetched
        ) : (
          <ChartComponent className="w-full h-full" data={data} predictData={predictData || []} />
        )}
      </div>

      {/* Radio Selections */}
      <div>
        {['lstnet', 'lstm', 'xgboost', 'mtgnn', 'random_forest', 'var', 'long-term'].map((option) => (
          <div key={option}>
            <label>
              <input
                type="radio"
                value={option}
                checked={algorithm === option}
                onChange={handleOptionChange}
              />
              {option}
            </label>
          </div>
        ))}
      </div>


      {/* Input */}
      <div>
        <input
          type="text"
          value={predictLength}
          onChange={handleInputChange}
          placeholder="Enter prediction length"
        />
      </div>

      {/* Button */}
      <div>
        <button onClick={handleButtonClick}>Execute API</button>
      </div>
    </main>
  );
}
