# Export the 544 AAindex1 amino-acid property scales distributed with SeqinR.
# Output: data/raw/physicochemical/aaindex1_seqinr.csv (already committed; rerun to regenerate).
# Usage: Rscript scripts/00_export_aaindex.R

suppressMessages(library(seqinr))
data(aaindex)

aa <- c("A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")
three <- c("Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
           "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val")

m <- t(sapply(aaindex, function(x) x$I[three]))
colnames(m) <- aa
out <- data.frame(index_id = names(aaindex), description = sapply(aaindex, function(x) x$D), m, check.names = FALSE)
write.csv(out, "data/raw/physicochemical/aaindex1_seqinr.csv", row.names = FALSE)
cat("Exported", nrow(m), "indices with seqinr", as.character(packageVersion("seqinr")), "\n")
