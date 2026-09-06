#!/bin/sh

exec gunicorn \
    --bind 0.0.0.0:8080 \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    serve:app