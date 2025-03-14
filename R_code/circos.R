
###########################################################################
prepare_data <- function(data_matrix, class_label, groups) {
  # Convert the matrix to a data frame
  df <- as.data.frame(data_matrix)

  # Correlation matrix
  corr_matrix <- cor(df)
  
  # Eigenvalues of the correlation matrix
  eigenvalues <- eigen(corr_matrix, symmetric = TRUE)$values
  
  # Effective number of tests
  M_eff <- sum(eigenvalues > 0)
  
  # Add class and group columns
  df$class <- class_label
  df$group <- groups
  
  # Calculate p-values between the two groups for each feature
  df_long <- df %>%
    pivot_longer(cols = -c(class, group), names_to = "feature", values_to = "expression")
  
  # Determine the number of unique groups
  num_groups <- length(unique(groups))
  
  # Calculate p-values depending on the number of groups
  p_values <- df_long %>%
    group_by(feature) %>%
    summarize(
      p_value = if (num_groups > 2) {
        # Perform ANOVA for more than 2 groups
        aov_result <- aov(expression ~ group)
        summary(aov_result)[[1]]$`Pr(>F)`[1]
      } else {
        # Perform t-test for exactly 2 groups
        wilcox.test(expression ~ group, paired = F, exact = F)$p.value
      },
      .groups = 'drop'
    ) %>%
    mutate(adjusted_p_value = p.adjust(p_value, method = "BH"))  # Adjust p-values
  
  
  # Pivot to long format
  df_long <- df %>%
    pivot_longer(
      cols = -c(class, group),  # Exclude 'class' and 'group' columns
      names_to = "feature",      # New column for feature names
      values_to = "expression"   # New column for expression values
    )
  
  # Summarize the mean expression
  data_summary <- df_long %>%
    group_by(group, feature, class) %>%
    summarize(mean_expression = mean(expression), .groups = 'drop')
  
  data_summary <- left_join(data_summary, p_values, by = "feature")
  
  return(data_summary)
}
###########################################################################
# Define the base path for the Measles data
base_path <- "School/Master 2/Thesis/Thesis/data/"

# Load CSV files (adjust the file paths to your own)
rna_data <- read.csv(paste0(base_path, "Measles/RNA_circos.csv"), row.names = 1)
metadata <- read.csv(paste0(base_path, "Measles/metadata.csv"), row.names = 1)
cytokine_data <- read.csv(paste0(base_path, "Measles/cytokines_data.csv"), row.names = 1)
cytometry_data <- read.csv(paste0(base_path, "Measles/cyto_data_sorted.csv"), row.names = 1)
cytokine_modules <- read.csv(paste0(base_path, "Measles/cytokine_modules.csv"), row.names = 1)
abTiters <- read.csv(paste0(base_path, "Measles/antibody_df.csv"))

# Construct a list similar to input.data
input.data <- list(
  X = list(
    RNA = rna_data,
    metadata = metadata,
    cytokineData = cytokine_data,
    cytometry = cytometry_data
  ),
  Y = list(
    abTiters = abTiters
  )
)

# Normalise the data
rna_data_normalized <- as.data.frame(scale(rna_data))
cytometry_normalized <- as.data.frame(scale(cytometry_data))

# Subset cytometry_data to keep only the desired columns
selected_columns <- c("WBC.Day.0", "RBC.Day.0", "PLT.Day.0", "X.LYM.Day.0", "X.MON.Day.0", "X.GRA.Day.0") # dropped "HGB.Day.0", "HCT.Day.0"
cytometry_normalized_subset <- cytometry_normalized[, selected_columns, drop = FALSE]

# Create block.pls.result with filtered RNA data and the cytometry subset
block.pls.result <- list(
  X = list(
    RNA = rna_data_normalized,
    cytometry = cytometry_normalized_subset
  )
)
###########################################################################

# Define the reference order based on one of the data frames' row names
reference_order <- rownames(input.data$Y$abTiters)

df_list <- list(input.data$X$RNA, 
                input.data$X$metadata,
                input.data$X$cytokineData, 
                input.data$X$cytometry,
                cytokine_modules)

# Update the row names to match the reference order
for (i in seq_along(df_list)) {
  rownames(df_list[[i]]) <- reference_order
}

# Update row names for each dataset in block.pls.result$X
for (i in seq_along(block.pls.result$X)) {
  rownames(block.pls.result$X[[i]]) <- reference_order
}

cytokine_modules <- df_list[[5]]


data_rna <- prepare_data(block.pls.result$X$RNA, 'RNA', input.data$Y$abTiters$quadrant)
data_rna <- data_rna[!grepl("TBD", data_rna$feature),]
data_cytokine <- prepare_data(cytokine_modules, 'Cytokines', input.data$Y$abTiters$quadrant)
data_cytometry <- prepare_data(block.pls.result$X$cytometry, 'Cytometry', input.data$Y$abTiters$quadrant)

# Combine the results
combined_data <- rbind(data_rna, data_cytokine, data_cytometry)

pivoted_data <- combined_data %>%
  pivot_wider(names_from = group, values_from = mean_expression)

p_value_threshold <- 0.25
color_significant <- "black"
color_non_significant <- "gray"
cor_threshold1 <- 0.75
cor_threshold2 <- 0.95

pivoted_data$color <- ifelse(pivoted_data$p_value < p_value_threshold,
                             color_significant, color_non_significant)

rna_df <- as.data.frame(block.pls.result$X$RNA[,!grepl("TBD", colnames(block.pls.result$X$RNA))])
rna_df <- rna_df[, sort(names(rna_df))]


#cytokine_df <- as.data.frame(block.pls.result$X$cytokine)
cytokine_df <- as.data.frame(cytokine_modules)
cytokine_df <- cytokine_df[, sort(names(cytokine_df))]
cytometry_df <- as.data.frame(block.pls.result$X$cytometry)
cytometry_df <- cytometry_df[, sort(names(cytometry_df))]

# Calculate column ranges for each data type
rna_cols <- ncol(rna_df)
cytokine_cols <- ncol(cytokine_df)
cytometry_cols <- ncol(cytometry_df)

# Function to determine sector based on index
get_sector <- function(index) {
  if (index <= rna_cols) {
    return("RNA")
  } else if (index <= rna_cols + cytokine_cols) {
    return("Cytokines")
  } else {
    return("Cytometry")
  }
}




### LINKS
#Combine dataframes by columns to calculate correlations
combined_df <- as.data.frame(cbind(rna_df,
                                   cytokine_df,
                                   cytometry_df))
# Calculate correlation matrix
correlation_matrix <- cor(combined_df, use = "pairwise.complete.obs")
diag(correlation_matrix) <- 0


plotsDir <- "School/Master 2/Thesis/Thesis/R_code/plots/"

### Draw the plot
png(filename = paste0(plotsDir,"circos_custom_previous.png"), width = 4000, height = 4000, res = 300)
circos.par("start.degree" = 90, "gap.degree" = 10)
circos.initialize(pivoted_data$class,
                  x = as.numeric(rownames(pivoted_data)))


circos.track(pivoted_data$class, 
             y = pivoted_data$responder,
             #y = pivoted_data$`peak response`,
             panel.fun = function(x, y) {
               # Add y-axis
               circos.yaxis(labels.cex = 0.6)  # Adjust size of y-axis labels
               
               circos.text(CELL_META$xcenter, 
                           CELL_META$cell.ylim[2] + mm_y(15), 
                           CELL_META$sector.index,
                           #facing = "clockwise", 
                           niceFacing = TRUE,
                           cex = 1.5
                          )
             })


circos.trackLines(pivoted_data$class, as.numeric(rownames(pivoted_data)),
                  #combined_data$mean_expression, col = combined_data$color, pch = 16, cex = 0.5)
                  pivoted_data$responder, col = "darkorange", pch = 16, cex = 0.5)
                  #pivoted_data$`peak response`, col = "darkorange", pch = 16, cex = 0.5)
circos.trackPoints(pivoted_data$class, as.numeric(rownames(pivoted_data)),
                  #combined_data$mean_expression, col = combined_data$color, pch = 16, cex = 0.5)
                  pivoted_data$responder, col = "darkorange", pch = 16, cex = 0.5)
                  #pivoted_data$`peak response`, col = "darkorange", pch = 16, cex = 0.5)

circos.trackLines(pivoted_data$class, as.numeric(rownames(pivoted_data)),
                  pivoted_data$`no response - high ab`, col = "darkblue", pch = 16, cex = 0.5)
                  #pivoted_data$`no response`, col = "darkblue", pch = 16, cex = 0.5)
                  #pivoted_data$`long response`, col = "darkblue", pch = 16, cex = 0.5)
circos.trackPoints(pivoted_data$class, as.numeric(rownames(pivoted_data)),
                  pivoted_data$`no response - high ab`, col = "darkblue", pch = 16, cex = 0.5)
                  #pivoted_data$`no response`, col = "darkblue", pch = 16, cex = 0.5)
                  #pivoted_data$`long response`, col = "darkblue", pch = 16, cex = 0.5)

### Additional
circos.trackLines(pivoted_data$class, as.numeric(rownames(pivoted_data)),
                  #pivoted_data$`high AB`, col = "darkgreen", pch = 16, cex = 0.5)
                  pivoted_data$`no response - low ab`, col = "darkgreen", pch = 16, cex = 0.5)
circos.trackPoints(pivoted_data$class, as.numeric(rownames(pivoted_data)),
                   #pivoted_data$`high AB`, col = "darkgreen", pch = 16, cex = 0.5)
                   pivoted_data$`no response - low ab`, col = "darkgreen", pch = 16, cex = 0.5)


#circos.trackLines(pivoted_data$class, as.numeric(rownames(pivoted_data)),
#                  pivoted_data$`low AB`, col = "purple3", pch = 16, cex = 0.5)
#circos.trackPoints(pivoted_data$class, as.numeric(rownames(pivoted_data)),
#                   pivoted_data$`low AB`, col = "purple3", pch = 16, cex = 0.5)



# Loop through the correlation matrix to add links
for (i in 1:(nrow(correlation_matrix))) {
  for (j in 1:(ncol(correlation_matrix))) {
    correlation <- correlation_matrix[i, j]
  
      
      # Define the link color based on correlation sign
      link_color <- if (correlation > 0) "red" else "blue"
      
      # Determine the sectors for i and j
      sector.index1 <- get_sector(i)
      sector.index2 <- get_sector(j)
      
      if (sector.index1 == "RNA") {
        rou1 = 0.5
      } else {
        rou1 = 0.62
      }
      if (sector.index2 == "RNA") {
        rou2 = 0.5
      } else {
        rou2 = 0.62
      }
      # Map i and j to their positions within their respective sectors
      pos1 <- i
      pos2 <- j
      
      if ((sector.index1 == sector.index2) && (abs(correlation) >= cor_threshold2)) {
        # Add a link within the same sector if correlation meets the threshold
        circos.link(sector.index1, pos1, sector.index2, pos2, col = link_color, 
                    rou1 = rou1, rou2 = rou2, lwd = .2)
        
      } else if ((sector.index1 != sector.index2) && (abs(correlation) >= cor_threshold1)) {
        # Add a link between different sectors if adjusted correlation meets the threshold
        circos.link(sector.index1, pos1, sector.index2, pos2, col = link_color, 
                    rou1 = rou1, rou2 = rou2, lwd = 1)
      }
  }
}



circos.labels(pivoted_data$class, x = as.numeric(rownames(pivoted_data)),
              col = pivoted_data$color,
              labels = pivoted_data$feature, cex = .62, side = 'inside', niceFacing = T)

legend("topright", legend = c("responder", "no response - high ab", "no response - low ab"), 
       col = c("darkorange", "darkblue", "darkgreen"), pch = 16, cex = 1.2, bty = "n")

circos.info()
dev.off()
circos.clear()