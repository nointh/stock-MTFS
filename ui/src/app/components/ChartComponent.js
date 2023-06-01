"use client";
import { createChart, ColorType } from 'lightweight-charts';
import React, { useState, useEffect, useRef } from 'react';

const ChartComponent = props => {
	const {
		data,
		className,
		colors: {
			backgroundColor = 'white',
			lineColor = '#2962FF',
			textColor = 'black',
			areaTopColor = '#2962FF',
			areaBottomColor = 'rgba(41, 98, 255, 0.28)',
		} = {},
		predictData,
	} = props;
	const [currentData, setCurrentData] = useState(null)
	const [isCurrentDataVisible, setCurrentDataVisible] = useState(true)
	const [chartType, setChartType] = useState('line')
	const chartContainerRef = useRef();
	const handleDataPanelCheckbox = event => {
		setCurrentDataVisible(event.target.checked)
	}
	const handleChartTypeInput = event => {
		let chartType = event.target.value;
		if (chartType !== 'area' && chartType !== 'candle') {
			chartType = 'line'
		}
		setChartType(chartType)
	}
	useEffect(
		() => {
			const allData = [...data, ...predictData];
			console.log("allData: ",allData)

			// Sort allData by time (date)
			allData.sort((a, b) => a.time.localeCompare(b.time));

			// Remove duplicates based on time
			const uniqueData = [];
			const uniqueDates = new Set();
			allData.forEach((element) => {
				const date = element.time.substring(0, 10);
				if (!uniqueDates.has(date)) {
					uniqueDates.add(date);
					uniqueData.push(element);
				}
			});

			const handleResize = () => {
				chart.applyOptions({ width: chartContainerRef.current.clientWidth });
			};

			const chart = createChart(chartContainerRef.current, {
				layout: {
					background: { type: ColorType.Solid, color: backgroundColor },
					textColor,
				},
				autoSize: true,
				height: 500,
			});
			chart.applyOptions({
				rightPriceScale: {
					visible: true,
				},
				leftPriceScale: {
					visible: true,
				},
			});
			chart.timeScale().fitContent();

			const priceSeries = chart.addLineSeries({ lineColor, topColor: areaTopColor, bottomColor: areaBottomColor, lineWidth: 2, priceScaleId: 'right', visible: chartType == 'line' });
			priceSeries.setData(uniqueData.map(
				element => {
					return { 'time': element['time'], 'value': element['close'] }
				}));

			const candleSeries = chart.addCandlestickSeries(
				{ lineColor, topColor: areaTopColor, bottomColor: areaBottomColor, lineWidth: 2, visible: chartType === 'candle' });
			candleSeries.setData(uniqueData.map(
				element => {
					return {
						...element, 'time': element['time']
					}
				}
			));
			const areaSeries = chart.addAreaSeries(
				{ lineColor, topColor: areaTopColor, bottomColor: areaBottomColor, lineWidth: 2, visible: chartType === 'area' });
			areaSeries.setData(uniqueData.map(
				element => {
					return { 'time': element['time'], 'value': element['close'] }
				}));
			const volumeSeries = chart.addHistogramSeries({
				priceFormat: {
					type: 'volume',
				},
				priceScaleId: 'left',
			});
			volumeSeries.priceScale('left').applyOptions({
				scaleMargins: {
					top: 0.8,
					bottom: 0,
				},
			});
			volumeSeries.setData(uniqueData.map(
				element => {
					return {
						'time': element['time'],
						'value': element['volume'],
						'color': element['change'] > 0 ? 'green' : 'red'
					}
				}));

			const predictPriceSeries = chart.addLineSeries(
				{ lineColor: 'orange', topColor: areaTopColor, bottomColor: areaBottomColor, lineWidth: 2, visible: chartType === 'line' });

			predictPriceSeries.setData(predictData.map(
				element => {
					return { 'time': element['time'], 'value': element['close'], 'color': 'orange' }
				}))

			const predictAreaSeries = chart.addAreaSeries(
				{ lineColor: 'orange', topColor: 'rgb(255, 213, 97)', bottomColor: 'rgb(242, 217, 148)', lineWidth: 2, visible: chartType === 'area' });
			predictAreaSeries.setData(predictData.map(
				element => {
					return { 'time': element['time'], 'value': element['close'] }
				}));

			const predictCandleSeries = chart.addCandlestickSeries(
				{ lineColor: 'orange', topColor: areaTopColor, bottomColor: areaBottomColor, lineWidth: 2, visible: chartType === 'candle' });

			predictCandleSeries.setData(
				predictData.map(element => {
					return {
						...element
					}
				})
			)
			predictCandleSeries.setMarkers(
				predictData.map(element => {
					return {
						'time': element['time'],
						position: 'aboveBar',
						color: 'orange',
						shape: 'arrowDown',
						id: 'id4',
						text: 'Predict',
						size: 1,
					}
				})
			)
			const predictVolumeSeries = chart.addHistogramSeries({
				priceFormat: {
					type: 'volume',
				},
				priceScaleId: 'left',
			});
			predictVolumeSeries.priceScale('left').applyOptions({
				scaleMargins: {
					top: 0.8,
					bottom: 0,
				},
			});
			predictVolumeSeries.setData(predictData.map(
				element => {
					return {
						'time': element['time'],
						'value': element['volume'],
						'color': 'orange'
					}
				}));

			chart.subscribeCrosshairMove((param) => {
				const datapoint = uniqueData.find(point => point.time === param.time);
				if (datapoint && isCurrentDataVisible) {
					setCurrentData(datapoint);
				}
			});


			window.addEventListener('resize', handleResize);

			return () => {
				window.removeEventListener('resize', handleResize);

				chart.remove();
			};
		},
		[data, isCurrentDataVisible, chartType, backgroundColor, lineColor, textColor, areaTopColor, areaBottomColor]
	);

	return (
		<div className="w-full">
			<nav className='w-full p-3 flex align-middle justify-center bg-white '>
				<label className="hidden sm:inline-flex relative items-center cursor-pointer">
					<input type="checkbox" value="" className="sr-only peer" checked={isCurrentDataVisible} onChange={handleDataPanelCheckbox} />
					<div className="w-11 h-6 bg-gray-200 rounded-full peer peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
					<span className="mx-3 text-sm font-medium text-gray-900 dark:text-gray-300">Show detail panel</span>
				</label>
				<label className="relative inline-flex items-center cursor-pointer">
					<select id="small" className="block text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" onChange={handleChartTypeInput} defaultValue={'line'}>
						<option value="line">Line</option>
						<option value="candle">Candle</option>
						<option value="area">Area</option>
					</select>
					<span className="mx-3 text-sm font-medium text-gray-900 dark:text-gray-300">Chart type</span>
				</label>

			</nav>
			<div
				className={`${className} relative`}
			>
				{isCurrentDataVisible && currentData && <div className={`hidden sm:block absolute top-0 left-0 right-0 mx-auto z-40 p-3 bg-gray-400 bg-opacity-70 rounded-md w-1/6`}>
					<div className='flex justify-between'><div>Open</div> <div>{currentData['open']}</div></div>
					<div className='flex justify-between'><div>High</div> <div>{currentData['high']}</div></div>
					<div className='flex justify-between'><div>Close</div> <div>{currentData['close']}</div></div>
					<div className='flex justify-between'><div>Volume</div> <div>{currentData['volume']}</div></div>
					<div className='flex justify-between'><div>% Change</div> <div>{currentData['change']} %</div></div>
				</div>}

				<div className={`w-full h-full`} ref={chartContainerRef}>
				</div>
			</div>
		</div>
	);
};
export default ChartComponent