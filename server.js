const express = require('express');
const cors = require('cors');
const app = express();
const admin = require('firebase-admin');
const { initializeApp } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');

let db;

if (process.env.FIREBASE_PROJECT_ID) {
    const serviceAccountConfig = {
        projectId: process.env.FIREBASE_PROJECT_ID,
        clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
        privateKey: process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, '\n'),
    };

    if (!admin.apps.length) {
        initializeApp({
            credential: admin.credential.cert(serviceAccountConfig)
        });
    }

    db = getFirestore();
    console.log('Firebase Admin SDK initialized successfully.');

} else {
    console.error('CRITICAL: Firebase Environment Variables are missing. Database updates will fail.');
}

app.use(express.json());

// 2. CORS Configuration - MUST run before your routes!
app.use(cors({
  origin: 'https://credinews-frontend.vercel.app', 
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'], 
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

if (db) {
    app.get('/verify', async (req, res) => {
        const userId = req.query.id; 
        const verificationToken = req.query.token;

        if (!userId || !verificationToken) {
            return res.status(400).send('Verification link is incomplete.');
        }

        try {
            const userRef = db.collection('users').doc(userId);
            const userDoc = await userRef.get();

            // Check if user exists and token is valid
            if (!userDoc.exists || userDoc.data().verificationToken !== verificationToken) {
                return res
                    .status(401)
                    .set('Access-Control-Allow-Origin', '*') // CORS fix
                    .send('Invalid or expired verification link.');
            }
            
            // This is the FIX: Sets verified = true
            await userRef.update({
                verified: true, 
                verificationToken: admin.firestore.FieldValue.delete(),
                verifiedAt: admin.firestore.FieldValue.serverTimestamp()
            });

            // Success response with CORS fix
            return res
                .set('Access-Control-Allow-Origin', '*') 
                .send('<h1>✅ Success!</h1><p>Your account has been successfully verified.</p>');

        } catch (error) {
            // Log the actual error that prevented the database write
            console.error(`Verification failed for user ${userId}:`, error);
            return res.status(500).send('Verification failed due to a server error. Please try again.');
        }
    });
}
// --- Your Existing API Route (Original Code) ---

app.post('/api/poser/analyze_full', (req, res) => {
  // Your logic to call your Python services or handle the request
  res.json({ message: 'Analysis complete', result: 'Success from Express backend' });
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Node.js Server running on port ${PORT}`);
});