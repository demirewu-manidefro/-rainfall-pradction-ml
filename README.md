# Rainfall Prediction System

This is a professional web application for predicting rainfall using a Machine Learning model (Random Forest).
Currently, the application runs with **Mock Data** for UI/UX demonstration purposes.

## Project Structure
- `app.py`: Flask backend serving the application and API.
- `templates/`: HTML templates for the frontend.
- `static/`: CSS and Javascript files.
- `Dockerfile`: Configuration for Docker deployment.

## Setup and Running

### Method 1: Local Python
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python app.py
   ```
3. Open your browser at `http://localhost:5000`.

### Method 2: Docker
1. Build the image:
   ```bash
   docker build -t rainfall-app .
   ```
2. Run the container:
   ```bash
   docker run -p 5000:5000 rainfall-app
   ```

## Features
- **Responsive Design**: Works on Mobile and Desktop.
- **Glassmorphism UI**: Modern aesthetic.
- **Interactive Charts**: Results visualized using Chart.js.
