# Base Image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Debugging:
RUN ls -la /usr/src/app

# Install needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Use environment variable for port
ARG DEFAULT_PORT=8501
ENV PORT=${DEFAULT_PORT}

# Enable permissions for correct file handling in container
ENV STREAMLIT_SERVER_ENABLE_CORS=false

RUN mkdir -p ~/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
enableXsrfProtection=false\n\
port = ${PORT}\n\
maxUploadSize=1028\n\
" > ~/.streamlit/config.toml

RUN ls -la /usr/src/app

# Make port available to the world outside this container
EXPOSE ${PORT}

# Run streamlit when the container launches
CMD ["streamlit",  "run", "detection-app.py"]
