import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="flex items-center justify-between bg-blue-300 text-white py-4 px-6">
      <div>
        <Link href="/">
          <img src="./src/app/components/logo.png" alt="Logo" className="h-8" />
        </Link>
      </div>
      <div className="flex items-center space-x-4">
        <div>
          <input
            type="text"
            placeholder="Search"
            className="px-4 py-2 rounded-md border border-gray-300 focus:outline-none"
          />
        </div>
        <div className="space-x-4">
          <Link href="/pricing">
            <span className="text-gray">Pricing</span>
          </Link>
          <Link href="/news">
            <span className="text-gray">News</span>
          </Link>
          <button className="bg-blue-500 text-white py-2 px-4 rounded-md">Signin</button>
          <button className="bg-blue-500 text-white py-2 px-4 rounded-md">Signup</button>
        </div>
      </div>
    </nav>
  );
}
