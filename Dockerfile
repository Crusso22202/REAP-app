# Use the official R Shiny image
FROM rocker/shiny:latest

# Set the CRAN mirror globally for all R sessions
RUN echo 'options(repos = c(CRAN = "https://cran.rstudio.com/"))' >> /usr/local/lib/R/etc/Rprofile.site

# Install required R packages
RUN R -e "install.packages(c('shiny', 'shinydashboard', 'DT', 'plotly', 'xgboost'))"

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



