import express from "express";
import multer from "multer";
import fs from "fs";
import { SpeechClient } from "@google-cloud/speech";
import { GoogleGenAI, HarmBlockThreshold, HarmCategory } from "@google/genai";
import axios from "axios";
import dotenv from "dotenv";
import cors from "cors";
dotenv.config();

const app = express();
const port = process.env.PORT || 3000;
app.use(cors());


const upload = multer({ dest: "/tmp" }); 

const speechClient = new SpeechClient({
  credentials: JSON.parse(process.env.GOOGLE_CLOUD_KEY),
});


async function transcribeAudio(filePath) {
  const audioContent = fs.readFileSync(filePath).toString("base64");

  const request = {
    audio: { content: audioContent },
    config: {
      encoding: "WEBM_OPUS",
      languageCode: "es-CO",
    },
  };

  try {
    const response = await speechClient.recognize(request);
    const transcription = response[0].results
      ? response[0].results.map((r) => r.alternatives[0].transcript).join(" ")
      : "";
    return transcription;
  } catch (error) {
    console.error("Error during transcription:", error);
    throw error;
  }
}

async function fetchProducts() {
  const res = await axios.get("https://trista-backend.vercel.app/products");
  return res.data.map((p) => ({ _id: p._id, name: p.name }));
}

async function callGemini(userInput, productsList) {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

  const config = {
    temperature: 0.1,
    maxOutputTokens: 1000,
    thinkingConfig: { thinkingBudget: 0 },
    safetySettings: [
      {
        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
      },
    ],
    systemInstruction: [
      {
        text: `
You are given a user input describing products with quantities in natural language. 
You also have a list of available products with _id.

Your task:
- Match the products mentioned in the user input to the available products.
- Return a JSON array of objects with the following format:
  [{ "_id": <product_id>, "count": <quantity> }]
- If a product is not mentioned or not found, it should not appear.
- If words are similar, match the most similar one.
- DO NOT return anything else besides valid JSON.

Products list: ${JSON.stringify(productsList)}
        `,
      },
    ],
  };

  const contents = [{ role: "user", parts: [{ text: userInput }] }];

  const response = await ai.models.generateContentStream({
    model: "gemini-flash-lite-latest",
    config,
    contents,
  });

  let output = "";
  for await (const chunk of response) {
    output += chunk.text;
  }

  try {
    return JSON.parse(output);
  } catch {
    console.error("Failed to parse Gemini response:", output);
    return [];
  }
}

app.post("/process-audio", upload.single("audio"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "Audio file missing" });

  try {
    // 1. Transcribir audio
    const transcription = await transcribeAudio(req.file.path);

    // 2. Fetch products
    const productsList = await fetchProducts();

    // 3. Llamar a Gemini
    const products = await callGemini(transcription, productsList);

    // 4. Limpiar archivo subido
    fs.unlinkSync(req.file.path);

    // 5. Devolver JSON al frontend
    res.json({ transcription, products });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
