import Image from 'next/image'
import ChartComponent from '../components/ChartComponent'
export default async function ChartPage() {
  const data = await getData()
  return (
    <main className="flex min-h-screen flex-col items-center justify-between md:p-24">

      {/* Chart Area */}
      <div className='z-10 w-full md:max-w-8xl max-h-[80%] items-center justify-between font-mono text-sm lg:flex'>
        <ChartComponent className="w-full h-full" data={data}
        ></ChartComponent>
      </div>


    </main>
  )
}

async function getData() {
  // Fetch data from external API
  const res = await fetch(`https://v30.ftisu.vn/get-vn30-history/`);
  const data = await res.json();
  const formatedData = data.map(element => {
    return {
      'time': new Date(element['Date']).toISOString().substring(0, 10),
      'open': element['Open'],
      'close': element['Price'],
      'high': element['High'],
      'low': element['Low'],
      'volume': element['Vol'],
      'change': element['Change'],
    }
  }).sort((a,b) => 
    Number(new Date(a['time'])) - Number(new Date(b['time']))
  )
  console.log(formatedData)
  // 
  // Pass data to the page via props
  return formatedData;
}
