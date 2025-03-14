workingDir = "School/Master 2/Thesis/Thesis/"

library(DESeq2)
library(tidyverse)

dataDir = paste0(workingDir, "data/")

###################################################################### Intro steps
# Load data
readcounts <- read.table(paste0(dataDir,"Hepatitis B/readcounts.csv"), sep = ',', header=T, row.names=1)
gene <- read.delim(paste0(dataDir, "Hepatitis B/genes.txt"),sep = "\t", header = FALSE, stringsAsFactors = FALSE)
colnames(gene) <- c("id", "col2","col3",  "name", "description", "col6", "col7", "chr", "start", "end")
meta <- read.table(paste0(dataDir,"Hepatitis B/rnaseq_metadata.csv"), sep = ',', header=T)

print(gene[16627, ])


# Remove unwanted rows
readcounts <- readcounts[!(rownames(readcounts) %in% c("_ambiguous", "_unmapped", "_no_feature")), ]

# Move to using gene names (from a precontructed txt file) and group genes with the same name
readcounts <- merge(x = readcounts,
                    y = gene[, c("id", "name")],
                    by.x = 0,            # use readcounts' row names
                    by.y = "id",         # match against gene file's id column
                    all = FALSE)
rownames(readcounts) <- readcounts$Row.names
readcounts <- readcounts[, !names(readcounts) %in% "Row.names"]
readcounts <- aggregate(. ~ name, data = readcounts, FUN = sum)
rownames(readcounts) <- readcounts$name
readcounts <- readcounts[, !names(readcounts) %in% "name"]
filtered_readcounts <- readcounts

###################################################################### Normalization
dds <- DESeqDataSetFromMatrix(countData = filtered_readcounts,
                              colData = meta,
                              design= ~ 1)
dds <- DESeq(dds)

vsd <- vst(dds, blind=TRUE)
vsd_counts <- assay(vsd)
vsd_counts <- as.data.frame(vsd_counts)


###################################################################### Average the replicates

transposed_readcounts <- t(vsd_counts)
transposed_readcounts <- as.data.frame(transposed_readcounts)
transposed_readcounts$prefix <- sub("^(H\\d+_EXP\\d+).*", "\\1", rownames(transposed_readcounts))

 
# Group by the prefix and calculate the mean
df_collapsed <- transposed_readcounts %>%
  group_by(prefix) %>%
  summarize(across(everything(), mean, na.rm = TRUE))
df_collapsed <- as.data.frame(df_collapsed)
rownames(df_collapsed) <- df_collapsed$prefix
df_collapsed <- df_collapsed %>% select(-prefix)

df_collapsed <- t(df_collapsed)
df_collapsed <- as.data.frame(df_collapsed)


write.csv(t(df_collapsed), paste0(dataDir, 'Hepatitis B/processed_readcounts_collapsed.csv'), row.names = T)