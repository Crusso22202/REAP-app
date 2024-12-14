# Use the official R Shiny image
FROM rocker/shiny:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev

# Copy app files to the Shiny Server directory
COPY ./ /srv/shiny-server/

# Expose the Shiny Server port
EXPOSE 3838

# Start Shiny Server
CMD ["/usr/bin/shiny-server"]



