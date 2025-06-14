library(WGCNA)
library(flashClust)
library(jsonlite)

produce_cytokine_modules <- function(data, plotTitle, plotsDir){
  
  require(pheatmap)
  require(ggplot2)
  require(glue)
  require(WGCNA)
  
  # Generate correlation matrix
  
  mat <- cor(data)
  
  # Create heatmap with pheatmap
  pheatmap_obj <- pheatmap(as.matrix(mat),
                           color = colorRampPalette(c("blue", "white", "red"))(100),
                           main = "",
                           display_numbers = TRUE,          
                           number_format = "%.2f",          
                           fontsize = 10,
                           fontsize_number = 10,
                           cutree_rows = 10,
                           cutree_cols = 10,
                           cluster_rows = TRUE, 
                           cluster_cols = TRUE,
                           filename = paste0(plotsDir, glue("{plotTitle}_euclidean_distance.png")),
                           width = 20, height = 20)
  
  
  row_dendrogram <- pheatmap_obj$tree_row
  
  # Cut the dendrogram to form 36 clusters
  row_clusters <- cutree(row_dendrogram, k = 36)
  
  # Group feature names by their cluster number
  clusters_list <- split(names(row_clusters), row_clusters)
  
  # Rename the clusters: add "Cluster" as a prefix to each key
  names(clusters_list) <- paste0("cluster", names(clusters_list))
  
  # Convert the clusters to JSON format
  json_clusters <- toJSON(clusters_list, pretty = TRUE)
  
  # Print the JSON output to the terminal
  cat("Clusters in JSON format:\n", json_clusters, "\n")
  
  png(paste0(plotsDir, glue("{plotTitle}_features_clustering.png")), width = 1400, height = 1000, res = 150)
  plot(as.dendrogram(row_dendrogram), main = "{plotTitle} Feature Clustering", sub = "", xlab = "", 
       cex.lab = 1.5, cex.axis = 1.5, cex.main = 2)
  dev.off()
  
  row_clusters <- cutree(row_dendrogram, k = 10)
  
  png(paste0(plotsDir, glue("{plotTitle}_features_clustering_cut.png")), width = 2400, height = 1000, res = 150) # width = 1400, height = 1000,
  plot(as.dendrogram(row_dendrogram), main = glue("{plotTitle} Feature Clustering"), sub = "", xlab = "", 
       cex.lab = 1.5, cex.axis = 1.5, cex.main = 2)
  rect.hclust(row_dendrogram, k = 10, border = "red")
  dev.off()
  
  png(paste0(plotsDir, glue("{plotTitle}_features_clustering_cut_colors.png")), width = 1400, height = 1000, res = 150) # width = 2800, height = 2000,
  clusterColors <- labels2colors(row_clusters)
  
  # Plot dendrogram
  plotDendroAndColors(
    dendro = row_dendrogram,
    colors = clusterColors,
    groupLabels = "Clusters",
    main = glue("{plotTitle} Feature Clustering"),
    cex.main = 1.5,         # Main title size (adjust as needed)
    cex.lab = 0.9,          # Group label size ("Clusters") (adjust as needed)
    cex.dendroLabels = 0.5 # Size for dendrogram leaf labels (feature names) - adjust this
    # WGCNA functions might also have cex.axis or similar depending on the version/specific plot
  )
  dev.off()
  
  return(clusterColors)
  
}

correlation_heatmap <- function(data, plotTitle, plotsDir){
  
  require(pheatmap)
  require(glue)
  
  # Generate correlation matrix
  mat <- cor(data)
  
  # Create heatmap with pheatmap
  pheatmap(as.matrix(mat),
           color = colorRampPalette(c("blue", "white", "red"))(100),
           main = "",
           display_numbers = TRUE,          
           number_format = "%.2f",          
           fontsize = 10,
           fontsize_number = 10,
           cutree_rows = 7,
           cutree_cols = 7,
           cluster_rows = TRUE, 
           cluster_cols = TRUE,
           filename = paste0(plotsDir, glue("{plotTitle}_correlation_plot.png")),
           width = 20, height = 20)
  
  return
  
}

test = read.csv("/Users/eliasdams/School/Master 2/Thesis/Thesis/data/cytokines_data_copy.csv")
# Remove the 'Vaccinee' and 'response_label' columns
test <- test[, !names(test) %in% c("Vaccinee", "response_label")]

produce_cytokine_modules(as.data.frame(test), "Cytokines", "School/Master 2/Thesis/Thesis/R_code/dendograms/")
#correlation_heatmap(as.data.frame(test), "Cytometry", "School/Master 2/Thesis/Thesis/R_code/plots/")