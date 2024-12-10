#!/bin/sh

# Start the Gunicorn server
exec gunicorn main:app --bind 0.0.0.0:${PORT:-5000}