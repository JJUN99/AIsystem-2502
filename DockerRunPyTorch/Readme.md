## Basic Running of Docker

# First, install and open Docker

# Build!
docker build --no-cache -t runptapi -f Docker/Dockerfile .

# Run
docker run -d -p 5003:5000 runptapi  

# See your local
http://localhost:5003/docs


