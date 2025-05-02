/**
 * Advanced WhatsApp AI Bot with Gemini API
 * Optimized for scalability and performance on Oracle Cloud Free Tier
 * Includes image processing capabilities with Arabic language support
 * Rate-limited to comply with Gemini API Free Tier limits (15 RPM, 1,500 RPD)
 */

// Required dependencies
const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const winston = require('winston');
const path = require('path');
const fs = require('fs');
const { promisify } = require('util');
const writeFileAsync = promisify(fs.writeFile);
const mkdirAsync = promisify(fs.mkdir);
const existsAsync = promisify(fs.exists);
const unlinkAsync = promisify(fs.unlink);
require('dotenv').config();

// Setup logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.printf(({ level, message, timestamp }) => {
      return `${timestamp} ${level}: ${message}`;
    })
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ],
});

// Initialize Express app with security enhancements
const app = express();
const PORT = process.env.PORT || 3000;

// Apply security middleware
app.use(helmet());
app.use(express.json({ limit: '1mb' }));

// Set up rate limiting to prevent abuse
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  standardHeaders: true,
  legacyHeaders: false,
});
app.use('/api', apiLimiter);

// Function to ensure media directory exists
async function ensureMediaDirExists() {
  const mediaDir = path.join(process.cwd(), 'media');
  if (!(await existsAsync(mediaDir))) {
    await mkdirAsync(mediaDir, { recursive: true });
  }
  return mediaDir;
}

// Initialize Gemini AI with error handling
let genAI, textModel, visionModel;
try {
  genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  // Initialize both models - one for text and one for images
  textModel = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
  // Update to use gemini-1.5-flash for vision tasks instead of the deprecated gemini-pro-vision
  visionModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
  logger.info('Gemini AI models initialized successfully');
} catch (error) {
  logger.error(`Failed to initialize Gemini AI: ${error.message}`);
  process.exit(1);
}

// Enhanced API Rate Limiter for Gemini API
// This enforces the free tier limits of 15 RPM and 1,500 RPD
class GeminiRateLimiter {
  constructor() {
    // Per-minute rate limiting
    this.requestsThisMinute = 0;
    this.minuteResetTime = Date.now() + 60000;
    
    // Per-day rate limiting
    this.requestsToday = 0;
    this.dayResetTime = this.calculateNextMidnight();
    
    // Set up interval to log usage stats
    setInterval(() => this.logUsageStats(), 10 * 60 * 1000); // Log every 10 minutes
  }
  
  calculateNextMidnight() {
    const now = new Date();
    const midnight = new Date(
      now.getFullYear(),
      now.getMonth(),
      now.getDate() + 1,
      0, 0, 0
    );
    return midnight.getTime();
  }
  
  async checkLimit() {
    const now = Date.now();
    
    // Check if we need to reset minute counter
    if (now > this.minuteResetTime) {
      this.requestsThisMinute = 0;
      this.minuteResetTime = now + 60000;
    }
    
    // Check if we need to reset daily counter
    if (now > this.dayResetTime) {
      this.requestsToday = 0;
      this.dayResetTime = this.calculateNextMidnight();
      logger.info(`Daily API request counter reset. New day started.`);
    }
    
    // Check if we've hit minute limit (leaving a buffer of 1 for safety)
    if (this.requestsThisMinute >= 14) { // 15 RPM limit - 1 for safety
      const waitTime = this.minuteResetTime - now;
      logger.warn(`Minute rate limit approached (${this.requestsThisMinute}/15). Waiting ${waitTime}ms before next request.`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      
      // After waiting, recursively check limits again
      return this.checkLimit();
    }
    
    // Check if we're approaching the daily limit (leave a buffer of 50 for safety)
    if (this.requestsToday >= 1450) { // 1500 RPD limit - 50 for safety
      const percentUsed = (this.requestsToday / 1500) * 100;
      logger.warn(`Daily API request limit at ${percentUsed.toFixed(1)}% (${this.requestsToday}/1500). Consider implementing fallback responses.`);
      
      // If we're very close to the limit, enable emergency conservation mode
      if (this.requestsToday >= 1490) {
        logger.error(`CRITICAL: Daily API request limit almost reached (${this.requestsToday}/1500). Using emergency conservation mode.`);
        return false; // Don't allow the request
      }
    }
    
    // We're within limits, allow the request
    return true;
  }
  
  async increment() {
    this.requestsThisMinute++;
    this.requestsToday++;
    
    // Log when hitting certain thresholds
    if (this.requestsToday % 100 === 0) {
      const percentUsed = (this.requestsToday / 1500) * 100;
      logger.info(`API daily usage: ${this.requestsToday}/1500 (${percentUsed.toFixed(1)}% of daily limit)`);
    }
  }
  
  logUsageStats() {
    const minutePercent = (this.requestsThisMinute / 15) * 100;
    const dailyPercent = (this.requestsToday / 1500) * 100;
    
    logger.info(`API Usage Stats - Minute: ${this.requestsThisMinute}/15 (${minutePercent.toFixed(1)}%), Day: ${this.requestsToday}/1500 (${dailyPercent.toFixed(1)}%)`);
    
    // Calculate time until reset
    const minutesUntilReset = ((this.minuteResetTime - Date.now()) / 1000 / 60).toFixed(1);
    const hoursUntilDayReset = ((this.dayResetTime - Date.now()) / 1000 / 60 / 60).toFixed(1);
    
    logger.info(`Rate limits reset in: ${minutesUntilReset} minutes (minute limit), ${hoursUntilDayReset} hours (daily limit)`);
  }
  
  getRemainingRequests() {
    return {
      minute: 15 - this.requestsThisMinute,
      day: 1500 - this.requestsToday
    };
  }
}

// Create the Gemini rate limiter instance
const geminiRateLimiter = new GeminiRateLimiter();

// User session management for context awareness
const userSessions = new Map();

// Clean up old sessions to prevent memory leaks
setInterval(() => {
  const now = Date.now();
  for (const [userId, session] of userSessions.entries()) {
    if (now - session.lastActivity > 3600000) { // 1 hour inactivity
      userSessions.delete(userId);
    }
  }
}, 600000); // Check every 10 minutes

// Pre-defined responses for when API limit is reached
const fallbackResponses = {
  arabic: {
    limitReached: "عذراً، لقد وصلنا إلى الحد الأقصى من الطلبات اليومية. يرجى المحاولة مرة أخرى غداً.",
    genericResponse: "أهلاً! أنا متاح حالياً بشكل محدود. سأرد على استفسارك قريباً عندما تتوفر الموارد.",
    imageAnalysis: "أستطيع رؤية الصورة التي أرسلتها، لكنني لا أستطيع تحليلها الآن بسبب قيود الاستخدام. يرجى المحاولة لاحقاً."
  },
  english: {
    limitReached: "Sorry, we've reached the maximum daily requests limit. Please try again tomorrow.",
    genericResponse: "Hello! I'm currently available in limited capacity. I'll respond to your query soon when resources are available.",
    imageAnalysis: "I can see the image you sent, but I can't analyze it right now due to usage restrictions. Please try again later."
  }
};

// Message queue to prevent overloading the API and handle rate limits
class MessageQueue {
  constructor() {
    this.queue = [];
    this.processing = false;
    this.rateLimit = {
      maxRequestsPerMinute: 14, // Set slightly below the 15 RPM Gemini limit
      requestsThisMinute: 0,
      resetTime: Date.now() + 60000
    };
  }

  async add(message) {
    return new Promise((resolve, reject) => {
      this.queue.push({
        message,
        resolve,
        reject
      });
      
      // Start processing if not already doing so
      if (!this.processing) {
        this.process();
      }
    });
  }

  async process() {
    if (this.queue.length === 0) {
      this.processing = false;
      return;
    }

    this.processing = true;
    
    // Check internal rate limit
    const now = Date.now();
    if (now > this.rateLimit.resetTime) {
      this.rateLimit.requestsThisMinute = 0;
      this.rateLimit.resetTime = now + 60000;
    }
    
    if (this.rateLimit.requestsThisMinute >= this.rateLimit.maxRequestsPerMinute) {
      // Wait until rate limit resets
      const waitTime = this.rateLimit.resetTime - now;
      logger.info(`Internal rate limit reached, waiting ${waitTime}ms`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      this.rateLimit.requestsThisMinute = 0;
      this.rateLimit.resetTime = Date.now() + 60000;
    }
    
    const task = this.queue.shift();
    this.rateLimit.requestsThisMinute++;
    
    try {
      // Process the message with error handling and retries
      const result = await this.processMessage(task.message);
      task.resolve(result);
    } catch (error) {
      logger.error(`Error processing queued message: ${error.message}`);
      task.reject(error);
    }
    
    // Small delay to prevent resource exhaustion
    setTimeout(() => this.process(), 100);
  }
  
  async processMessage(message) {
    let retries = 3;
    
    // Check if message has media to process
    let hasMedia = false;
    try {
      hasMedia = await message.hasMedia;
    } catch (error) {
      logger.error(`Error checking media: ${error.message}`);
      hasMedia = false;
    }
    
    // Get or create user session for this sender
    if (!userSessions.has(message.from)) {
      userSessions.set(message.from, {
        messageCount: 0,
        lastActivity: Date.now(),
        pendingResponse: false,
        language: 'ar', // Default to Arabic
        conversation: [] // Track conversation history
      });
    }
    
    const userSession = userSessions.get(message.from);
    userSession.lastActivity = Date.now();
    
    // Add the current message to conversation history
    userSession.conversation.push({
      role: 'user',
      content: message.body || (hasMedia ? 'صورة بدون نص' : '') // 'Image without text' in Arabic
    });
    
    // Keep only the last 10 messages to prevent context from getting too large
    if (userSession.conversation.length > 10) {
      userSession.conversation = userSession.conversation.slice(-10);
    }
    
    // Check if we can make an API call based on rate limits
    const canMakeRequest = await geminiRateLimiter.checkLimit();
    
    // If we can't make a request due to rate limits, provide a fallback response
    if (!canMakeRequest) {
      const fallbackResponse = fallbackResponses.arabic.limitReached;
      
      // Save the fallback response to conversation history for context
      userSession.conversation.push({
        role: 'assistant',
        content: fallbackResponse
      });
      
      return fallbackResponse;
    }
    
    while (retries > 0) {
      try {
        // Handle image processing if message has media
        if (hasMedia) {
          logger.info(`Processing message with media from ${message.from.split('@')[0]}`);
          
          try {
            // Get media from the message
            const media = await message.downloadMedia();
            
            // Check if it's an image
            if (media && media.mimetype && media.mimetype.startsWith('image/')) {
              // Create media directory if it doesn't exist
              const mediaDir = await ensureMediaDirExists();
              
              // Create unique file name based on timestamp
              const fileName = `${Date.now()}-${Math.floor(Math.random() * 10000)}.jpg`;
              const imagePath = path.join(mediaDir, fileName);
              
              // Save media to disk
              const imageBuffer = Buffer.from(media.data, 'base64');
              await writeFileAsync(imagePath, imageBuffer);
              
              // Log image details
              logger.info(`Image saved to ${imagePath}, size: ${imageBuffer.length} bytes`);
              
              // Get caption or default text
              const prompt = message.body || "اشرح هذه الصورة بالتفصيل"; // Default Arabic prompt
              
              // Prepare image for Gemini Vision API
              const imageData = {
                inlineData: {
                  data: media.data,
                  mimeType: media.mimetype
                }
              };
              
              // Create a system prompt to ensure Arabic responses
              const systemPrompt = "أنت مساعد ذكي يتحدث باللغة العربية فقط. قم بتحليل الصورة والرد بالعربية الفصحى بشكل مفصل ومفيد."; // Arabic system prompt
              
              // Call Gemini Vision API with instruction to respond in Arabic
              const result = await visionModel.generateContent({
                contents: [{
                  parts: [
                    { text: systemPrompt + "\n\n" + prompt },
                    imageData
                  ]
                }]
              });
              
              // Increment API usage counter
              await geminiRateLimiter.increment();
              
              // Delete temporary file
              try {
                await unlinkAsync(imagePath);
                logger.info(`Temporary image deleted: ${imagePath}`);
              } catch (err) {
                logger.warn(`Failed to delete temp image: ${err.message}`);
              }
              
              const response = result.response.text();
              
              // Save the bot's response to conversation history
              userSession.conversation.push({
                role: 'assistant',
                content: response
              });
              
              return response;
            } else {
              const response = "يمكنني تحليل الصور فقط. يرجى إرسال صورة بتنسيق JPG أو PNG أو WebP.";
              
              // Save the bot's response to conversation history
              userSession.conversation.push({
                role: 'assistant',
                content: response
              });
              
              return response;
            }
          } catch (mediaError) {
            logger.error(`Error processing media: ${mediaError.message}`);
            return "واجهت مشكلة في معالجة صورتك. يرجى إعادة إرسالها أو تجربة تنسيق مختلف.";
          }
        } else {
          // Process text-only message
          const conversationContext = userSession.conversation
            .slice(-6) // Use last 6 messages for context
            .map(msg => `${msg.role === 'user' ? 'المستخدم' : 'المساعد'}: ${msg.content}`)
            .join('\n');
          
          // Build prompt with conversation history and instruction to respond in Arabic
          const fullPrompt = `
            أنت مساعد ذكي يتحدث باللغة العربية فقط. يجب أن تكون جميع ردودك باللغة العربية.
            
            المحادثة السابقة:
            ${conversationContext}
            
            سؤال المستخدم الحالي: ${message.body}
            
            قدم إجابة مفيدة ومفصلة باللغة العربية:
          `;
          
          const result = await textModel.generateContent({
            contents: [{
              parts: [{ text: fullPrompt }]
            }]
          });
          
          // Increment API usage counter
          await geminiRateLimiter.increment();
          
          const response = result.response.text();
          
          // Save the bot's response to conversation history
          userSession.conversation.push({
            role: 'assistant',
            content: response
          });
          
          return response;
        }
      } catch (error) {
        retries--;
        logger.error(`API call failed, retries left: ${retries}. Error: ${error.message}`);
        
        if (retries === 0) throw error;
        
        // Exponential backoff
        await new Promise(resolve => setTimeout(resolve, (4 - retries) * 1000));
      }
    }
  }
}

const messageQueue = new MessageQueue();

// Create a WhatsApp client instance with enhanced configuration
const client = new Client({
  authStrategy: new LocalAuth({
    dataPath: path.join(process.cwd(), '.wwebjs_auth')
  }),
  puppeteer: {
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--no-first-run',
      '--no-zygote',
      '--disable-gpu',
      '--disable-extensions',
      '--disable-background-networking',
      '--disable-default-apps',
      '--disable-sync',
      '--disable-translate',
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-component-extensions-with-background-pages',
      '--disable-ipc-flooding-protection',
      '--enable-features=NetworkService',
      '--mute-audio',
    ],
    ignoreHTTPSErrors: true,
  }
});

// Session management state
let qrCodeText = null;
let isClientReady = false;

// Handle QR code generation with security enhancements
client.on('qr', (qr) => {
  qrCodeText = qr;
  logger.info('QR CODE GENERATED. Scan with your phone.');
  qrcode.generate(qr, { small: true });
});

// Create a secure QR code endpoint
app.get('/qr', (req, res) => {
  const apiKey = req.query.key;
  
  // Simple API key auth for the QR endpoint
  if (apiKey !== process.env.ADMIN_API_KEY) {
    logger.warn(`Unauthorized QR code access attempt from ${req.ip}`);
    return res.status(401).send('Unauthorized');
  }
  
  if (qrCodeText) {
    res.send(`
      <html>
        <head>
          <title>WhatsApp Bot QR Code</title>
          <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
            .container { max-width: 500px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .qr-container { margin: 20px 0; }
          </style>
        </head>
        <body>
          <div class="container">
            <h1>WhatsApp Bot QR Code</h1>
            <p>Scan this code with your WhatsApp to authenticate</p>
            <div class="qr-container">
              <pre>${qrCodeText}</pre>
            </div>
            <p>This code will expire after use</p>
          </div>
        </body>
      </html>
    `);
  } else {
    res.send('No QR code available. Bot may already be authenticated.');
  }
});

// Handle client ready event
client.on('ready', () => {
  isClientReady = true;
  qrCodeText = null; // Clear QR code after authentication
  logger.info('WhatsApp client authenticated and ready');
});

// Handle authentication failures with reconnection logic
client.on('auth_failure', (msg) => {
  isClientReady = false;
  logger.error(`Authentication failure: ${msg}`);
  
  // Attempt to reconnect with exponential backoff
  setTimeout(() => {
    logger.info('Attempting to reconnect...');
    client.initialize();
  }, 5000);
});

// Handle disconnects with robust reconnection
client.on('disconnected', (reason) => {
  isClientReady = false;
  logger.warn(`Client disconnected: ${reason}`);
  
  // Implement exponential backoff for reconnection
  const reconnect = (attempt = 1) => {
    const delay = Math.min(Math.pow(2, attempt) * 1000, 60000); // Max 1 minute delay
    logger.info(`Reconnection attempt ${attempt} in ${delay}ms`);
    
    setTimeout(() => {
      if (!isClientReady) {
        try {
          client.initialize();
          // If not reconnected within 30 seconds, try again
          setTimeout(() => {
            if (!isClientReady) {
              reconnect(attempt + 1);
            }
          }, 30000);
        } catch (error) {
          logger.error(`Reconnection attempt failed: ${error.message}`);
          reconnect(attempt + 1);
        }
      }
    }, delay);
  };
  
  reconnect();
});

// Process incoming messages with enhanced error handling and rate limiting
client.on('message', async (message) => {
  const timestamp = new Date().toISOString();
  const sender = message.from.split('@')[0];
  
  // Check if message has media
  let hasMedia = false;
  try {
    hasMedia = await message.hasMedia;
  } catch (error) {
    logger.error(`Error checking media: ${error.message}`);
  }
  
  // Log differently based on message type
  if (hasMedia) {
    logger.info(`[${timestamp}] Media message received from ${sender}`);
  } else {
    logger.info(`[${timestamp}] Message from ${sender}: ${message.body}`);
  }
  
  try {
    // Ignore group messages
    if (message.from.includes('@g.us')) {
      return;
    }
    
    // Update or create user session
    if (!userSessions.has(message.from)) {
      userSessions.set(message.from, {
        messageCount: 0,
        lastActivity: Date.now(),
        pendingResponse: false,
        language: 'ar', // Default to Arabic
        conversation: [] // Track conversation history
      });
    }
    
    const userSession = userSessions.get(message.from);
    userSession.lastActivity = Date.now();
    userSession.messageCount++;
    
    // Check if we already have a pending response to prevent spamming
    if (userSession.pendingResponse) {
      logger.info(`Already processing a message for ${sender}, skipping`);
      return;
    }
    
    userSession.pendingResponse = true;
    
    // Use the message queue to process the message
    try {
      const text = await messageQueue.add(message);
      
      // Break long responses into multiple messages for reliability
      if (text.length > 4000) {
        const chunks = splitTextIntoChunks(text, 4000);
        for (const chunk of chunks) {
          await message.reply(chunk);
          // Add a small delay between messages
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      } else {
        await message.reply(text);
      }
      
      logger.info(`Response sent successfully to ${sender}`);
    } catch (error) {
      logger.error(`Error processing message from ${sender}: ${error.message}`);
      await message.reply('عذراً، واجهت خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى بعد قليل.'); // Arabic error message
    } finally {
      userSession.pendingResponse = false;
    }
  } catch (error) {
    logger.error(`Unexpected error handling message: ${error.message}`);
  }
});

// Helper function to split long messages
function splitTextIntoChunks(text, maxLength) {
  const chunks = [];
  let currentChunk = "";
  
  // Split by paragraphs first to maintain formatting
  const paragraphs = text.split("\n");
  
  for (const paragraph of paragraphs) {
    if (currentChunk.length + paragraph.length + 1 > maxLength) {
      // If adding this paragraph exceeds max length, finalize current chunk
      chunks.push(currentChunk);
      currentChunk = paragraph + "\n";
    } else {
      // Otherwise add paragraph to current chunk
      currentChunk += paragraph + "\n";
    }
  }
  
  // Add the last chunk if it has content
  if (currentChunk.trim()) {
    chunks.push(currentChunk);
  }
  
  return chunks;
}

// Cleanup temporary files periodically
setInterval(async () => {
  try {
    const mediaDir = path.join(process.cwd(), 'media');
    if (await existsAsync(mediaDir)) {
      fs.readdir(mediaDir, async (err, files) => {
        if (err) {
          logger.error(`Error reading media directory: ${err.message}`);
          return;
        }
        
        const now = Date.now();
        for (const file of files) {
          try {
            const filePath = path.join(mediaDir, file);
            const stats = fs.statSync(filePath);
            // Delete files older than 1 hour
            if (now - stats.mtimeMs > 3600000) {
              await unlinkAsync(filePath);
              logger.info(`Deleted old temp file: ${file}`);
            }
          } catch (fileErr) {
            logger.error(`Error processing file ${file}: ${fileErr.message}`);
          }
        }
      });
    }
  } catch (error) {
    logger.error(`Error in temp file cleanup: ${error.message}`);
  }
}, 3600000); // Run every hour

// Enhanced health endpoint to include API usage stats
app.get('/health', (req, res) => {
  const apiUsage = {
    requestsPerMinute: geminiRateLimiter.requestsThisMinute,
    requestsPerDay: geminiRateLimiter.requestsToday,
    minuteLimit: 15,
    dailyLimit: 1500,
    remainingMinute: 15 - geminiRateLimiter.requestsThisMinute,
    remainingDaily: 1500 - geminiRateLimiter.requestsToday,
    minuteResetIn: ((geminiRateLimiter.minuteResetTime - Date.now()) / 1000).toFixed(1) + " seconds",
    dayResetIn: ((geminiRateLimiter.dayResetTime - Date.now()) / (1000 * 60 * 60)).toFixed(1) + " hours"
  };
  
  res.json({
    status: 'ok',
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    clientReady: isClientReady,
    activeUsers: userSessions.size,
    queueSize: messageQueue.queue.length,
    apiUsage: apiUsage
  });
});

// System status monitoring
app.get('/', (req, res) => {
  res.send('WhatsApp Bot Server is running!');
});

// Start the Express server
const server = app.listen(PORT, () => {
  logger.info(`Server is running on port ${PORT}`);
});

// Graceful server shutdown
process.on('SIGTERM', gracefulShutdown);
process.on('SIGINT', gracefulShutdown);

async function gracefulShutdown() {
  logger.info('Received shutdown signal, closing connections...');
  
  // Close the Express server
  server.close(() => {
    logger.info('HTTP server closed');
  });
  
  // Destroy WhatsApp client
  try {
    await client.destroy();
    logger.info('WhatsApp client destroyed');
  } catch (error) {
    logger.error(`Error destroying WhatsApp client: ${error.message}`);
  }
  
  // Exit process after a timeout
  setTimeout(() => {
    logger.info('Exiting process');
    process.exit(0);
  }, 3000);
}

// Initialize the WhatsApp client
try {
  client.initialize();
  logger.info('WhatsApp client initialization started');
} catch (error) {
  logger.error(`Error initializing WhatsApp client: ${error.message}`);
  process.exit(1);
}