# Stop all running containers
#docker stop $(docker ps -q)

# Remove all containers
#docker rm $(docker ps -a -q)

# Remove unused Docker images
#docker image prune -a -f

docker build -t oil-palm-detect .
docker run -p 8501:8501 oil-palm-detect