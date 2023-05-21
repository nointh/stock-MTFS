"use client"
import React, { useState, useEffect } from 'react';
import ChartComponent from '../components/ChartComponent';


export default function ChartPage() {
  const [selectedOption, setSelectedOption] = useState('');
  const [inputValue, setInputValue] = useState('');
  const [data, setData] = useState();
  const [isLoading, setIsLoading] = useState(true);
  const [predictData, setPredictData] = useState();

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
    const apiUrl = `https://ec2-13-239-176-190.ap-southeast-2.compute.amazonaws.com/predict/lstnet`;
    fetch(apiUrl)
      .then((response) => response.json())
      .then((data) => {
        // Handle the API response data
        const formattedData = data.data.VN30.reduce((accumulator, element) => {
          const date = element.date;
          const existingItem = accumulator.find((item) => item.time === date);
          if (!existingItem) {
            const formattedElement = {
              open: element.open || 0,
              close: element.close || 0,
              high: element.high || 0,
              low: element.low || 0,
              volume: element.volume || 0,
              time: element.date || 0,
              change: element.change || 0,
            };
            accumulator.push(formattedElement);
          }
          return accumulator;
        }, []);
        console.log("predict Data", formattedData);
        setPredictData(formattedData); // Update the predictData state with the formatted data
      })
      .catch((error) => {
        // Handle errors
        console.error(error);
      });
  };
  



  async function fetchData() {
    try {
      const res = await fetch('https://ec2-13-239-176-190.ap-southeast-2.compute.amazonaws.com/history');
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
  
      console.log(formattedData[2317]);
      console.log(formattedData[2316]);
      console.log("history res", formattedData);
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
