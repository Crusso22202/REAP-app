install.packages('reticulate')

# app.R
library(shiny)
library(Matrix)
library(shinydashboard)
library(reticulate)
library(scales)
library(DT)
library(plotly)
library(xgboost)
library(caret)
reticulate::py_install("xgboost")
reticulate::py_install("scikit-learn")

#Model----
data <- `ICRE_df_model.(1)`

# Start with data selection to match Python exactly
df_model <- data[c(numerical_features, categorical_features, "Sale.Price")]

# Do train/test split first
set.seed(42)
train_index <- createDataPartition(df_model$Sale.Price, p = 0.8, list = FALSE)
train_data <- df_model[train_index, ]
test_data <- df_model[-train_index, ]

# One-hot encode categorical variables with drop_first=TRUE
cat_formula <- as.formula(paste("~", paste(categorical_features, collapse = " + ")))  # Removed -1
train_dummies <- model.matrix(cat_formula, data = train_data)[, -1]  # -1 removes intercept (equivalent to drop_first)
test_dummies <- model.matrix(cat_formula, data = test_data)[, -1]

# Scale numerical features after split
scaler <- preProcess(train_data[numerical_features], method = c("center", "scale"))
train_data_scaled <- predict(scaler, train_data[numerical_features])
test_data_scaled <- predict(scaler, test_data[numerical_features])

# Combine features
train_matrix <- cbind(as.matrix(train_data_scaled), train_dummies)
test_matrix <- cbind(as.matrix(test_data_scaled), test_dummies)
colnames(test_matrix) <- colnames(train_matrix)

# Train model with unscaled target
xgb_model <- xgboost(
  data = train_matrix,
  label = train_data$Sale.Price,  # Unscaled target
  params = list(
    objective = "reg:squarederror",
    max_depth = 3,
    eta = 0.05,
    min_child_weight = 7,
    subsample = 0.5,
    colsample_bytree = 0.5,
    gamma = 0.5,
    alpha = 0.1,
    lambda = 0.1
  ),
  nrounds = 100,
  verbose = 0
)

# Calculate metrics using unscaled predictions
train_pred <- predict(xgb_model, train_matrix)
test_pred <- predict(xgb_model, test_matrix)

# Calculate metrics to match Python's
train_r2 <- 1 - sum((train_data$Sale.Price - train_pred)^2) / 
  sum((train_data$Sale.Price - mean(train_data$Sale.Price))^2)
test_r2 <- 1 - sum((test_data$Sale.Price - test_pred)^2) / 
  sum((test_data$Sale.Price - mean(test_data$Sale.Price))^2)

train_mse <- mean((train_data$Sale.Price - train_pred)^2)
test_mse <- mean((test_data$Sale.Price - test_pred)^2)

train_mae <- mean(abs(train_data$Sale.Price - train_pred))
test_mae <- mean(abs(test_data$Sale.Price - test_pred))

print("Training Metrics:")
print(paste("  Mean Squared Error:", format(train_mse, scientific = TRUE)))
print(paste("  R-squared:", round(train_r2, 4)))
print(paste("  Mean Absolute Error:", round(train_mae, 4)))

print("\nTesting Metrics:")
print(paste("  Mean Squared Error:", format(test_mse, scientific = TRUE)))
print(paste("  R-squared:", round(test_r2, 4)))
print(paste("  Mean Absolute Error:", round(test_mae, 4)))

# Save everything needed for predictions
saveRDS(list(
  model = xgb_model,
  scaler = scaler,
  categorical_formula = cat_formula,
  dummy_names = dummy_names,
  feature_names = colnames(train_matrix),
  numerical_features = numerical_features
), "industrial_re_model.rds")

#Dashboard----

# Load the saved model and objects
model_objects <- readRDS("industrial_re_model.rds")

# Define feature sets
numerical_features <- c(
  "Building.SF", "Year.Built", "Building.Tax.Expenses", 
  "Coverage", "Improvement.Ratio", "Distance_to_Nearest_Port",
  "Latitude", "Longitude", "distance_public_transport",
  "distance_to_highway", "distance_to_nearest_airport",
  "distance_to_nearest_rail", "distance_to_nearest_dc",
  "distance_to_residential", "Market.Asking.Rent.Growth",
  "Vacancy.Rate", "Sales.Volume.Transactions", "Market.Cap.Rate",
  "Under.Construction.SF", "Net.Absorption.SF", "Population",
  "Industrial.Employment", "Under.Construction.SF.Growth",
  "Population.Growth", "Industrial.Employment.Growth"
)

categorical_features <- c("Secondary.Type", "Building.Class", "Sale.Quarter")

# UI Definition
ui <- dashboardPage(
  dashboardHeader(title = "Industrial Real Estate Predictor"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Price Prediction", tabName = "prediction", icon = icon("calculator")),
      menuItem("Model Info", tabName = "model_info", icon = icon("info-circle"))
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper, .right-side {
          background-color: #f4f6f9;
        }
        .box {
          box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .box-header {
          background-color: #f8f9fa;
        }
      "))
    ),
    
    tabItems(
      # Prediction Tab
      tabItem(
        tabName = "prediction",
        fluidRow(
          box(
            title = "Property Details", status = "primary", solidHeader = TRUE,
            width = 6,
            numericInput("building_sf", "Building Square Footage",
                         value = 100000, min = 0),
            numericInput("year_built", "Year Built",
                         value = 2000, min = 1900, max = 2024),
            numericInput("building_tax", "Building Tax Expenses",
                         value = 50000, min = 0),
            numericInput("coverage", "Coverage",
                         value = 0.5, min = 0, max = 1, step = 0.01),
            numericInput("improvement_ratio", "Improvement Ratio",
                         value = 0.5, min = 0, max = 1, step = 0.01),
            
            selectInput("secondary_type", "Secondary Type",
                        choices = c("Distribution", "Food Processing", "Freestanding",
                                    "Light Distribution", "Light Manufacturing",
                                    "Manufacturing", "R&D", "Refrigeration/Cold Storage",
                                    "Service", "Showroom", "Telecom Hotel/Data Hosting",
                                    "Truck Terminal", "Warehouse")),
            
            selectInput("building_class", "Building Class",
                        choices = c("A", "B", "C", "F")),
            
            selectInput("sale_quarter", "Sale Quarter",
                        choices = c("2020Q1", "2020Q2", "2020Q3", "2020Q4",
                                    "2021Q1", "2021Q2", "2021Q3", "2021Q4",
                                    "2022Q1", "2022Q2", "2022Q3", "2022Q4",
                                    "2023Q1", "2023Q2", "2023Q3", "2023Q4",
                                    "2024Q1", "2024Q2", "2024Q3", "2024Q4"))
          ),
          
          box(
            title = "Location Details", status = "primary", solidHeader = TRUE,
            width = 6,
            numericInput("latitude", "Latitude",
                         value = 40.7128, min = -90, max = 90),
            numericInput("longitude", "Longitude",
                         value = -74.0060, min = -180, max = 180),
            numericInput("distance_port", "Distance to Nearest Port (miles)",
                         value = 10, min = 0),
            numericInput("distance_transport", "Distance to Public Transport (miles)",
                         value = 2, min = 0),
            numericInput("distance_highway", "Distance to Highway (miles)",
                         value = 1, min = 0),
            numericInput("distance_airport", "Distance to Nearest Airport",
                         value = 5, min = 0),
            numericInput("distance_rail", "Distance to Nearest Rail",
                         value = 3, min = 0),
            numericInput("distance_dc", "Distance to Nearest DC",
                         value = 4, min = 0),
            numericInput("distance_residential", "Distance to Residential",
                         value = 2, min = 0)
          )
        ),
        
        fluidRow(
          box(
            title = "Market Conditions", status = "primary", solidHeader = TRUE,
            width = 6,
            numericInput("market_rent_growth", "Market Asking Rent Growth (%)",
                         value = 2, min = -100, max = 100),
            numericInput("vacancy_rate", "Vacancy Rate (%)",
                         value = 5, min = 0, max = 100),
            numericInput("sales_volume", "Sales Volume Transactions",
                         value = 100, min = 0),
            numericInput("market_cap_rate", "Market Cap Rate (%)",
                         value = 6, min = 0, max = 100),
            numericInput("under_construction", "Under Construction SF",
                         value = 50000, min = 0),
            numericInput("net_absorption", "Net Absorption SF",
                         value = 10000),
            numericInput("population", "Population",
                         value = 100000, min = 0),
            numericInput("industrial_employment", "Industrial Employment",
                         value = 10000, min = 0),
            numericInput("under_construction_growth", "Under Construction SF Growth",
                         value = 0),
            numericInput("population_growth", "Population Growth",
                         value = 0),
            numericInput("industrial_employment_growth", "Industrial Employment Growth",
                         value = 0)
          ),
          
          box(
            title = "Prediction Results", status = "success", solidHeader = TRUE,
            width = 6,
            actionButton(
              "calculate",
              "Calculate Price",
              class = "btn-success",
              style = "margin-bottom: 15px; width: 100%;"
            ),
            htmlOutput("prediction_output"),
            plotlyOutput("feature_importance", height = "300px")
          )
        )
      ),
      
      # Model Info Tab
      tabItem(
        tabName = "model_info",
        fluidRow(
          box(
            title = "Model Performance",
            width = 12,
            solidHeader = TRUE,
            status = "info",
            HTML("
              <div style='padding: 20px;'>
                <h4>Model Overview</h4>
                <p>This model uses XGBoost (eXtreme Gradient Boosting) to predict industrial real estate prices
                based on multiple features including property characteristics, location details, and market conditions.</p>
                
                <h4>Model Metrics:</h4>
                <ul>
                  <li>Training R² Score: 0.917</li>
                  <li>Testing R² Score: 0.793</li>
                  <li>Mean Absolute Error: $2,439,426</li>
                </ul>
                
                <h4>Feature Categories:</h4>
                <ul>
                  <li>Property Characteristics: Building size, age, class, type, etc.</li>
                  <li>Location Factors: Distance to transport, ports, highways, etc.</li>
                  <li>Market Conditions: Vacancy rates, rent growth, employment, etc.</li>
                </ul>
                
                <h4>Usage Notes:</h4>
                <p>Predictions are most accurate when input values fall within typical market ranges.
                Extreme values may result in less reliable predictions.</p>
              </div>
            ")
          )
        )
      )
    )
  )
)

# Server Logic
server <- function(input, output, session) {
  # Define all possible levels for categorical variables
  secondary_type_levels <- c(
    "Distribution", "Food Processing", "Freestanding", "Light Distribution", 
    "Light Manufacturing", "Manufacturing", "R&D", "Refrigeration/Cold Storage",
    "Service", "Showroom", "Telecom Hotel/Data Hosting", "Truck Terminal", 
    "Warehouse", "Contractor Storage Yard"
  )
  
  building_class_levels <- c("A", "B", "C", "F")
  
  sale_quarter_levels <- c(
    "2019Q4", "2020Q1", "2020Q2", "2020Q3", "2020Q4",
    "2021Q1", "2021Q2", "2021Q3", "2021Q4",
    "2022Q1", "2022Q2", "2022Q3", "2022Q4",
    "2023Q1", "2023Q2", "2023Q3", "2023Q4",
    "2024Q1", "2024Q2", "2024Q3", "2024Q4"
  )
  
  prepare_input_data <- function(input) {
    # Handle numeric features
    numeric_data <- data.frame(
      "Building.SF" = as.numeric(input$building_sf),
      "Year.Built" = as.numeric(input$year_built),
      "Building.Tax.Expenses" = as.numeric(input$building_tax),
      "Coverage" = as.numeric(input$coverage),
      "Improvement.Ratio" = as.numeric(input$improvement_ratio),
      "Distance_to_Nearest_Port" = as.numeric(input$distance_port),
      "Latitude" = as.numeric(input$latitude),
      "Longitude" = as.numeric(input$longitude),
      "distance_public_transport" = as.numeric(input$distance_transport),
      "distance_to_highway" = as.numeric(input$distance_highway),
      "distance_to_nearest_airport" = as.numeric(input$distance_airport),
      "distance_to_nearest_rail" = as.numeric(input$distance_rail),
      "distance_to_nearest_dc" = as.numeric(input$distance_dc),
      "distance_to_residential" = as.numeric(input$distance_residential),
      "Market.Asking.Rent.Growth" = as.numeric(input$market_rent_growth),
      "Vacancy.Rate" = as.numeric(input$vacancy_rate),
      "Sales.Volume.Transactions" = as.numeric(input$sales_volume),
      "Market.Cap.Rate" = as.numeric(input$market_cap_rate),
      "Under.Construction.SF" = as.numeric(input$under_construction),
      "Net.Absorption.SF" = as.numeric(input$net_absorption),
      "Population" = as.numeric(input$population),
      "Industrial.Employment" = as.numeric(input$industrial_employment),
      "Under.Construction.SF.Growth" = as.numeric(input$under_construction_growth),
      "Population.Growth" = as.numeric(input$population_growth),
      "Industrial.Employment.Growth" = as.numeric(input$industrial_employment_growth)
    )
    
    # Scale numeric features
    scaled_numeric <- predict(model_objects$scaler, numeric_data)
    
    # Create categorical data frame with predefined levels
    cat_data <- data.frame(
      Secondary.Type = factor(input$secondary_type, levels = secondary_type_levels),
      Building.Class = factor(input$building_class, levels = building_class_levels),
      Sale.Quarter = factor(input$sale_quarter, levels = sale_quarter_levels)
    )
    
    # Create dummy variables
    cat_formula <- as.formula("~ Secondary.Type + Building.Class + Sale.Quarter - 1")
    cat_matrix <- model.matrix(cat_formula, data = cat_data)
    
    # Combine numeric and categorical features
    final_matrix <- matrix(0, nrow = 1, ncol = length(model_objects$feature_names))
    colnames(final_matrix) <- model_objects$feature_names
    
    # Fill in values
    for(col in colnames(scaled_numeric)) {
      if(col %in% model_objects$feature_names) {
        final_matrix[1, col] <- scaled_numeric[1, col]
      }
    }
    
    for(col in colnames(cat_matrix)) {
      if(col %in% model_objects$feature_names) {
        final_matrix[1, col] <- cat_matrix[1, col]
      }
    }
    
    return(as.matrix(final_matrix))
  }
  
  # Create a reactive value to store the prediction result
  prediction_result <- reactiveVal(NULL)
  
  # Update prediction only when calculate button is clicked
  observeEvent(input$calculate, {
    req(input$building_sf, input$year_built)
    
    tryCatch({
      input_matrix <- prepare_input_data(input)
      
      print("Matrix structure:")
      print(dim(input_matrix))
      print(head(colnames(input_matrix)))
      
      pred <- predict(model_objects$model, input_matrix)
      prediction_result(pred)
      
    }, error = function(e) {
      print("Error details:")
      print(e)
      prediction_result(NULL)
    })
  })
  
  # Render prediction output based on stored prediction value
  output$prediction_output <- renderUI({
    if (is.null(prediction_result())) {
      if (input$calculate == 0) {
        # Initial state, before first calculation
        HTML(paste0(
          '<div style="text-align: center; padding: 20px;">',
          '<p style="color: #7f8c8d;">Click "Calculate Price" to get the prediction</p>',
          '</div>'
        ))
      } else {
        # Error state
        HTML(paste0(
          '<div style="color: #c0392b; padding: 20px; text-align: center;">',
          '<h3>Error in prediction</h3>',
          '<p>Please check your inputs and try again.</p>',
          '</div>'
        ))
      }
    } else {
      # Success state
      formatted_price <- paste0("$", format(round(prediction_result()), big.mark = ",", scientific = FALSE))
      HTML(paste0(
        '<div style="text-align: center; padding: 20px;">',
        '<h2 style="color: #2c3e50;">Estimated Price</h2>',
        '<h1 style="color: #27ae60; margin: 20px 0;">', formatted_price, '</h1>',
        '<p style="color: #7f8c8d; font-style: italic;">',
        'Based on current market conditions and property characteristics',
        '</p></div>'
      ))
    }
  })
        
        # Feature importance plot remains the same
        output$feature_importance <- renderPlotly({
          importance_matrix <- xgb.importance(model = model_objects$model)
          
          plot_ly(importance_matrix, 
                  x = ~reorder(Feature, Gain),
                  y = ~Gain,
                  type = "bar",
                  marker = list(color = "#27ae60")) %>%
            layout(title = "Feature Importance",
                   xaxis = list(title = "Features", tickangle = 45),
                   yaxis = list(title = "Relative Importance"),
                   margin = list(b = 100))
        })
      }

# Run the application
shinyApp(ui = ui, server = server)

