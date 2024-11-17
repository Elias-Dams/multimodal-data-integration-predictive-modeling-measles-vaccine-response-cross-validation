library(WGCNA)
library(flashClust)

produce_cytokine_modules <- function(input_mat, plotsDir) {
  
  require(pheatmap)
  
  # Compute correlation matrix
  cor_matrix <- cor(input_mat, use = "pairwise.complete.obs")
  
  # Convert correlation matrix to a distance matrix for clustering
  dist_matrix <- as.dist(1 - abs(cor_matrix))
  
  # Hierarchical clustering
  hc <-  stats::hclust(dist_matrix, method = "ward.D2")
  
  
  png(paste0(plotsDir, "merged_features_clustering.png"), width = 1400, height = 1000, res = 150)
  plot(hc, main = "Cytokine Feature Clustering", sub = "", xlab = "", cex.lab = 1.5, cex.axis = 1.5, cex.main = 2)
  dev.off()
  
  png(paste0(plotsDir, "merged_features_clustering_cut.png"), width = 1400, height = 1000, res = 150)
  plot(hc, main = "Cytokine Feature Clustering", sub = "", xlab = "", cex.lab = 1.5, cex.axis = 1.5, cex.main = 2)
  rect.hclust(hc, h = .9, border = "red",)
  dev.off()
  
  png(paste0(plotsDir, "merged_features_clustering_cut_colors.png"), width = 1400, height = 1000, res = 150)
  clusters <- cutree(hc, h = .9)
  clusterColors <- labels2colors(clusters)
  
  # Plot dendrogram
  plotDendroAndColors(
    dendro = hc,
    colors = clusterColors,         
    groupLabels = "Clusters",       
    main = "Cytokine Feature Clustering"
  )
  dev.off()
  
  
  pheatmap(cor_matrix,
           cluster_rows = hc,        
           cluster_cols = hc,        
           display_numbers = TRUE,   
           number_format = "%.2f",   
           fontsize_number = 10,     
           cutree_rows = length(unique(clusters)),
           cutree_cols = length(unique(clusters)),
           main = "Clustered Correlation Heatmap",
           filename = paste0(plotsDir, "correlation_heatmap_merged_ward2.png"),
           width = 20, height = 20)
  
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

test = read.csv("School/Master 2/Thesis/Thesis/data/merged_cytokines_cyto_meta.csv")
# Remove the 'Vaccinee' and 'response_label' columns
test <- test[, !names(test) %in% c("Vaccinee", "response_label")]

produce_cytokine_modules(as.data.frame(test), "School/Master 2/Thesis/Thesis/R_code/")
correlation_heatmap(as.data.frame(test), "Cytokine", "School/Master 2/Thesis/Thesis/R_code/")