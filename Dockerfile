# Use the latest FastAI image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Debugging: List contents of /usr/src/app
RUN ls -la /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Create a Streamlit configuration directory and file
RUN mkdir -p ~/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
enableXsrfProtection=false\n\
port = 8501\n\
maxUploadSize=1028\n\
" > ~/.streamlit/config.toml

RUN ls -la /usr/src/app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit when the container launches
CMD ["streamlit",  "run", "detection-app.py"]
