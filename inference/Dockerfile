# Use Kubeflow PyTorch CPU image as base
FROM kubeflow/pytorch-torchrun-cpu:v1-52e41d5

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app and model
COPY app.py .
COPY model/ ./model/

# Expose FastAPI port
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]