// NEW FILE: server.js (in the root of your fly.io deployment folder)

const express = require('express');
const cors = require('cors');
const app = express();

// --- Core Middleware Setup ---

// 1. Required for parsing JSON bodies in POST requests
app.use(express.json());

// 2. CORS Configuration - MUST run before your routes!
app.use(cors({
  origin: 'https://credinews-frontend.vercel.app', // Your Vercel frontend URL
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'], // Include OPTIONS for preflight
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

// --- Your API Routes ---

app.post('/api/poser/analyze_full', (req, res) => {
  // Your logic to call your Python services or handle the request
  res.json({ message: 'Analysis complete', result: 'Success from Express backend' });
});

// ... other routes (if any) ...

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Node.js Server running on port ${PORT}`);
});