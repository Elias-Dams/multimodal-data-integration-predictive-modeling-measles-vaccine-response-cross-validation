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
                           fontsize = 5,
                           fontsize_number = 5,
                           cutree_rows = 36,
                           cutree_cols = 36,
                           cluster_rows = TRUE, 
                           cluster_cols = TRUE,
                           filename = paste0(plotsDir, glue("{plotTitle}_euclidean_distance.png")),
                           width = 40, height = 40)
  
  
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
  
  row_clusters <- cutree(row_dendrogram, k = 14)
  
  png(paste0(plotsDir, glue("{plotTitle}_features_clustering_cut.png")), width = 1400, height = 1000, res = 150) # width = 1400, height = 1000,
  plot(as.dendrogram(row_dendrogram), main = glue("{plotTitle} Feature Clustering"), sub = "", xlab = "", 
       cex.lab = 1.5, cex.axis = 1.5, cex.main = 2)
  rect.hclust(row_dendrogram, k = 14, border = "red")
  dev.off()
  
  png(paste0(plotsDir, glue("{plotTitle}_features_clustering_cut_colors.png")), width = 1400, height = 1000, res = 150) # width = 2800, height = 2000,
  clusterColors <- labels2colors(row_clusters)
  
  # Plot dendrogram
  plotDendroAndColors(
    dendro = row_dendrogram,
    colors = clusterColors,         
    groupLabels = "Clusters",       
    main = glue("{plotTitle} Feature Clustering")
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

test = read.csv("School/Master 2/Thesis/Thesis/data/Hepatitis B/RNA_circos.csv")
# Remove the 'Vaccinee' and 'response_label' columns
test <- test[, !names(test) %in% c("Vaccinee", "response_label")]

produce_cytokine_modules(as.data.frame(test), "RNA_Hepatitis B", "School/Master 2/Thesis/Thesis/R_code/plots/")
#correlation_heatmap(as.data.frame(test), "Cytometry", "School/Master 2/Thesis/Thesis/R_code/plots/")