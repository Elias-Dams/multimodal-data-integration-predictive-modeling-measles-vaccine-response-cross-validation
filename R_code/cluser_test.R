# Generate Modules --------------------------------------------------------
# Load WGCNA and flashClust libraries every time you open R
library(WGCNA)
library(flashClust)


# Uploading data into R and formatting it for WGCNA
# This creates an object called "datExpr" that contains the normalized counts file output from DESeq2
datExpr = read.csv("School/Master 2/Thesis/Thesis/data/cytokines_data_copy.csv")
# "head" the file to preview it
geneLabels = colnames(datExpr)
print(geneLabels)
datExpr <- log2(datExpr + 1)  # Simple log transformation for normalization
head(datExpr) # You see that genes are listed in a column named "X" and samples are in columns

# Manipulate file so it matches the format WGCNA needs
row.names(datExpr) = datExpr$X
datExpr$X = NULL
datExpr = as.data.frame(datExpr) # now samples are rows and genes are columns
dim(datExpr) # 48 samples and 1000 genes (you will have many more genes in reality)


# Run this to check if there are gene outliers
gsg = goodSamplesGenes(datExpr, verbose = 3)
gsg$allOK

# Choose a soft threshold power- USE A SUPERCOMPUTER IRL ------------------------------------a

powers = seq(from =10, to=30, by=1) #choosing a set of soft-thresholding powers
sft = pickSoftThreshold(datExpr, powerVector=powers, verbose =5, networkType="signed") #call network topology analysis function

sizeGrWindow(9,5)
par(mfrow= c(1,2))
cex1=0.7
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], xlab= "Soft Threshold (power)", ylab="Scale Free Topology Model Fit, signed R^2", type= "n", main= paste("Scale independence"))
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], labels=powers, cex=cex1, col="red")
abline(h=0.80, col="red")
abline(v=20, col="green")
plot(sft$fitIndices[,1], sft$fitIndices[,5], xlab= "Soft Threshold (power)", ylab="Mean Connectivity", type="n", main = paste("Mean connectivity"))
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1, col="red")


#build a adjacency "correlation" matrix
enableWGCNAThreads()
softPower = 1
adjacency = adjacency(datExpr, power = softPower, type = "signed") #specify network type
head(adjacency)

# Construct Networks- USE A SUPERCOMPUTER IRL -----------------------------
#translate the adjacency into topological overlap matrix and calculate the corresponding dissimilarity:
TOM = TOMsimilarity(adjacency, TOMType="signed") # specify network type
dissTOM = 1-TOM



# Generate a clustered gene tree
geneTree = flashClust(as.dist(dissTOM), method="average")
#This sets the minimum number of genes to cluster into a module
minModuleSize = 1
dynamicMods = cutreeDynamic(dendro= geneTree, distM= dissTOM, deepSplit=4, pamRespectsDendro= FALSE, minClusterSize = minModuleSize)
dynamicColors= labels2colors(dynamicMods)

#plot dendrogram with module colors below it
plotDendroAndColors(geneTree,  dynamicColors, "Clusters", hang = 0.03, dendroLabels = geneLabels,  addGuide = TRUE)

#INCLUE THE NEXT LINE TO SAVE TO FILE
# dev.off()