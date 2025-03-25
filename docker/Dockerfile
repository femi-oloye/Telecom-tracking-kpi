# Use an official lightweight Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements.txt from the parent directory
COPY ../requirements.txt . 

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Install Streamlit explicitly to ensure it is in $PATH
RUN pip install --no-cache-dir streamlit
RUN pip install --no-cache-dir mlflow


# Copy the entire project from the parent directory
COPY .. .

# Expose the port for Streamlit
EXPOSE 8502

# Default command: Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
