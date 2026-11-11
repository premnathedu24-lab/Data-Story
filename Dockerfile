FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd -m -u 1000 appuser
COPY --chown=appuser:appuser app.py analysis.py enhancements.py reporting.py model_guidance.py ml_engine.py ./
COPY --chown=appuser:appuser .streamlit/ .streamlit/
USER appuser
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false"]
