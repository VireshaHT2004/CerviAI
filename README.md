# CerviAI

CerviAI is a Flask-based web application for managing cervical cancer screening workflows. The project includes a doctor dashboard, technician dashboard, review workflow, and a backend prediction pipeline.

## Project structure

- api/ - Vercel-compatible entry point for the backend
- backend/ - Flask application, models, and prediction utilities
- frontend/ - frontend-related files and notes
- templates/ - HTML templates for the dashboards and auth pages
- static/ - static assets such as JavaScript and images
- data/ - inbox and output data folders
- database/ - database storage placeholder

## Features

- User authentication and registration
- Doctor dashboard
- Technician dashboard
- Review workflow for screening results
- Prediction support using backend utility scripts

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

## Running locally

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
python backend/app.py
```

The app will run locally on:

```text
http://127.0.0.1:5000
```

## Deployment

This project includes Vercel deployment configuration through:

- vercel.json
- api/index.py

## Notes

- The project uses Flask and Python-based backend services.
- Some files and folders such as model weights or large media assets may need to be added separately depending on your environment.

## License

This project is intended for educational and internal use unless otherwise specified.
