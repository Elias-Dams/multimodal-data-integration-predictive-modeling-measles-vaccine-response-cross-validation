
###########################################################################
# 1) Prepare_data function
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
# 2) Load data
###########################################################################
base_path <- "School/Master 2/Thesis/Thesis/data/"

# RNA data (row names = samples)
rna_data <- read.csv(paste0(base_path, "Measles/RNA_circos.csv"), row.names = 1)

# Antibody titers data, which must contain some grouping info (e.g., quadrant or response label)
abTiters <- read.csv(paste0(base_path, "Measles/antibody_df.csv"))

# If the grouping is stored in abTiters$quadrant, we can use that as 'groups'.
groups_vector <- abTiters$quadrant  # or whichever column indicates group

# normalize RNA data
rna_data_normalized <- as.data.frame(scale(rna_data))

###########################################################################
# 3) Prepare the data
###########################################################################
data_rna <- prepare_data(rna_data_normalized, 'RNA', groups_vector)

# If you want to remove features containing "TBD" or otherwise prune:
data_rna <- data_rna[!grepl("TBD", data_rna$feature), ]

# Pivot to wide format for plotting
pivoted_data <- data_rna %>%
  tidyr::pivot_wider(names_from = group, values_from = mean_expression)

###########################################################################
# 4) Determine color for p-values
###########################################################################
p_value_threshold <- 0.25  # more stringent threshold
color_significant <- "black"
color_non_significant <- "gray"

pivoted_data$color <- ifelse(pivoted_data$p_value < p_value_threshold,
                             color_significant, color_non_significant)


###########################################################################
# 6) Circos plot
###########################################################################
# We'll define a single sector = "RNA", so we can simplify
get_sector <- function(index) {
  return("RNA")
}

# Set thresholds for correlation links
cor_threshold <- 0.95

plotsDir <- "School/Master 2/Thesis/Thesis/R_code/plots/"
png(filename = paste0(plotsDir,"circos_RNA_only.png"), width = 4000, height = 4000, res = 300)

# Initialize circos with the row names of pivoted_data as the x-axis
circos.par("start.degree" = 90, "gap.degree" = 10)
circos.initialize(pivoted_data$class, x = as.numeric(rownames(pivoted_data)))

# Create a track for each group’s expression
# E.g., if you have "responder", "no response - high ab", "no response - low ab", etc.
# We'll assume pivoted_data columns have these group names
unique_groups <- setdiff(colnames(pivoted_data), c("feature","class","p_value","adjusted_p_value","color"))

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


# Next, draw correlation links
n_features <- ncol(rna_df)
for (i in seq_len(n_features)) {
  for (j in seq_len(n_features)) {
    correlation <- correlation_matrix[i, j]
    if (abs(correlation) < cor_threshold) next  # skip small correlations
    
    link_color <- if (correlation > 0) "red" else "blue"
    
    # Since there's only RNA data, we define a single sector
    sector <- "RNA"
    # Set a rou value that determines how far the link starts from the circle edge.
    # Adjust this value to position the links further out (e.g., rou = 0.55).
    rou <- 0.55
    
    circos.link(sector, i, sector, j, col = link_color, rou1 = rou, rou2 = rou, lwd = .2)
  }
}

# Optionally label features
circos.labels(pivoted_data$class, x = as.numeric(rownames(pivoted_data)),
              col = pivoted_data$color,
              labels = pivoted_data$feature, cex = 0.5, side = 'inside', niceFacing = TRUE)

# Add a legend if desired
legend("topright", legend = c("responder", "no response - high ab", "no response - low ab"), 
       col = c("darkorange", "darkblue", "darkgreen"), pch = 16, cex = 1.2, bty = "n")

circos.info()
dev.off()
circos.clear()