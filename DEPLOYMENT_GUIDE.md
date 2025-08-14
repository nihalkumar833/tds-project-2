# TDS Data Analyst - Railway Deployment Guide

## Project Setup Complete! ✅

The project has been set up with all necessary deployment configurations for Railway.

## Files Created

1. **`.env`** - Environment configuration file with placeholder for Google API key
2. **`Dockerfile`** - Container configuration for Railway deployment
3. **`railway.json`** - Railway-specific deployment configuration
4. **`Procfile`** - Process file for Railway deployment
5. **`runtime.txt`** - Python version specification
6. **`.dockerignore`** - Files to exclude from Docker build

## Deployment Instructions

### 1. Update Environment Variables

Before deploying, update your `.env` file with your actual Google API key:
```
GOOGLE_API_KEY=your_actual_google_api_key_here
```

### 2. Push to GitHub

```bash
cd /mnt/d/tds-project\ 2.0/Data-Analyst
git init
git add .
git commit -m "Initial commit with Railway deployment configuration"
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

### 3. Deploy to Railway

#### Option A: Deploy via Railway Dashboard
1. Go to [Railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway will automatically detect the configuration

#### Option B: Deploy via Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Link to GitHub repo
railway link

# Deploy
railway up
```

### 4. Configure Environment Variables on Railway

In Railway Dashboard:
1. Go to your project
2. Click on "Variables"
3. Add the following:
   - `GOOGLE_API_KEY`: Your Google Generative AI API key (REQUIRED)
   - Other variables are optional (defaults are set in railway.json)

### 5. Access Your Application

Once deployed, Railway will provide you with a URL like:
```
https://your-app-name.up.railway.app
```

## Local Testing

To test the application locally:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the application
uvicorn app:app --host 0.0.0.0 --port 8000

# Or with Python directly
python app.py
```

Access at: http://localhost:8000

## Docker Testing (Optional)

To test the Docker container locally:

```bash
# Build the image
docker build -t tds-data-analyst .

# Run the container
docker run -p 8000:8000 --env-file .env tds-data-analyst
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GOOGLE_API_KEY` | Google Generative AI API Key | - | Yes |
| `PORT` | Application port | 8000 | No |
| `MODEL_NAME` | AI Model to use | gemini-1.5-flash | No |
| `TEMPERATURE` | Model temperature | 0.7 | No |
| `MAX_TOKENS` | Max generation tokens | 2048 | No |
| `REQUEST_TIMEOUT` | Request timeout (seconds) | 60 | No |
| `API_TIMEOUT` | API timeout (seconds) | 30 | No |

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure all dependencies are in `requirements.txt`
2. **Port binding issues**: Railway automatically assigns a PORT environment variable
3. **API key errors**: Verify your Google API key is correctly set in Railway variables
4. **Build failures**: Check Docker logs in Railway dashboard

### Logs

View application logs in Railway:
```bash
railway logs
```

Or in the Railway dashboard under "Deployments" → "View Logs"

## Features

- 📊 Advanced data analysis with AI
- 📈 Interactive visualizations
- 🤖 Google Gemini integration
- 📁 Multiple file format support (CSV, Excel, JSON)
- 🔍 Automated insights generation
- 📊 Statistical analysis
- 🎨 Customizable charts and graphs

## Support

For issues or questions:
- Check the [original repository](https://github.com/22f3000359/Data-Analyst)
- Railway documentation: https://docs.railway.app
- Google AI documentation: https://ai.google.dev