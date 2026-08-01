# api/index.py
# Vercel Python entrypoint that wraps the Flask WSGI app
from vercel_wsgi import handle
from backend.app import app as flask_app

# Vercel expects a top-level callable named `handler(event, context)`
def handler(event, context):
    return handle(event, context, flask_app)
