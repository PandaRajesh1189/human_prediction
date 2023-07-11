# Use the official Python base image
FROM python:3.10.12

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .
COPY inception_model_pkl_3 .
COPY model_pkl .

# Install the required packages
RUN pip install -r requirements.txt

# Copy the app files to the working directory
COPY . .

# Expose the port that Streamlit will use
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
