"use client"
import React, { useEffect } from 'react';
import ChartComponent from '../components/ChartComponent';

export default function ChartPage() {
  const [selectedOption, setSelectedOption] =  React.useState('');
  const [inputValue, setInputValue] =  React.useState('');
  const [data, setData] =  React.useState([]);

  useEffect(() => {
    getData();
  }, []);

  const handleOptionChange = (event) => {
    setSelectedOption(event.target.value);
  };

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleButtonClick = () => {

    const apiUrl = `YOUR_API_ENDPOINT?selectedOption=${selectedOption}&inputValue=${inputValue}`;
    fetch(apiUrl)
      .then((response) => response.json())
      .then((data) => {
        // Handle the API response data
        console.log(data);
      })
      .catch((error) => {
        // Handle errors
        console.error(error);
      });
  };

  async function getData() {
    const res = await fetch('http://localhost:8000/get-vn30-history/');
    const data = await res.json();
    const formattedData = data
      .map((element) => ({
        time: new Date(element['Date']).toISOString().substring(0, 10),
        open: element['Open'],
        close: element['Price'],
        high: element['High'],
        low: element['Low'],
        volume: element['Vol'],
        change: element['Change'],
      }))
      .sort((a, b) => Number(new Date(a['time'])) - Number(new Date(b['time'])));
    setData(formattedData);
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between md:p-24">
      {/* Chart Area */}
      <div className="z-10 w-full md:max-w-8xl max-h-[80%] items-center justify-between font-mono text-sm lg:flex">
        <ChartComponent className="w-full h-full" data={data} />
      </div>

      {/* Radio Selections */}
      <div>
        <label>
          <input
            type="radio"
            value="option1"
            checked={selectedOption === 'option1'}
            onChange={handleOptionChange}
          />
          Option 1
        </label>
        <label>
          <input
            type="radio"
            value="option2"
            checked={selectedOption === 'option2'}
            onChange={handleOptionChange}
          />
          Option 2
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
