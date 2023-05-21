"use client"
import React, { useState, useEffect } from 'react';
import ChartComponent from '../components/ChartComponent';

export default function ChartPage() {
  const [selectedOption, setSelectedOption] = useState('');
  const [inputValue, setInputValue] = useState('');
  const [data, setData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [predictData, setPredictData] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const handleOptionChange = (event) => {
    setSelectedOption(event.target.value);
  };

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleButtonClick = () => {
    const apiUrl = `http://localhost:8000/api/predictions/?algorithm=${selectedOption}&predict_range=${inputValue}`;
    fetch(apiUrl)
      .then((response) => response.json())
      .then((data) => {
        // Handle the API response data
        console.log(data);
        setPredictData(data); // Update the predictData state with the received data
      })
      .catch((error) => {
        // Handle errors
        console.error(error);
      });
  };

  async function fetchData() {
    try {
      const res = await fetch('http://ec2-13-239-176-190.ap-southeast-2.compute.amazonaws.com/history');
      const data = await res.json();
      const formattedData = data.map((element) => ({
        time: new Date(element.date).toISOString().substring(0, 10),
        open: element.open,
        close: element.close,
        high: element.high,
        low: element.low,
        volume: element.volume,
        change: element.change,
      }));
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
        <label>
          <input
            type="radio"
            value="lstm"
            checked={selectedOption === 'lstm'}
            onChange={handleOptionChange}
          />
          lstm
        </label>
        <label>
          <input
            type="radio"
            value="xgboost"
            checked={selectedOption === 'xgboost'}
            onChange={handleOptionChange}
          />
          xgboost
        </label>
        {/* Add more radio options as needed */}
      </div>

      {/* Input */}
      <div>
        <input
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          placeholder="Enter a value"
        />
      </div>

      {/* Button */}
      <div>
        <button onClick={handleButtonClick}>Execute API</button>
      </div>
    </main>
  );
}
