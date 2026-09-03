#!/usr/bin/env Rscript

# Publication-resolution human–chimpanzee alignment plot for one inversion.
# Human (GRCh38) is always the upper target track. Chimpanzee (panTro6) is the
# lower query track, reoriented when necessary so the predominant alignment in
# the two flanks is forward. A shallow red box marks the inversion only on the
# human track; it does not imply exact orthologous chimpanzee breakpoints.

suppressPackageStartupMessages({
  library(SVbyEye)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8) {
  stop(
    "usage: plot_chimp_hires.R PAF OUT_PREFIX INV_ID CHROM INV_START ",
    "INV_END REGION_OFFSET LABEL"
  )
}

paf_file <- args[[1]]
out_prefix <- args[[2]]
inv_id <- args[[3]]
chrom <- args[[4]]
inv_start <- as.numeric(args[[5]])
inv_end <- as.numeric(args[[6]])
region_offset <- as.numeric(args[[7]])
label <- args[[8]]

inv_length <- inv_end - inv_start
if (!is.finite(inv_length) || inv_length <= 0) {
  stop("INV_END must be greater than INV_START")
}

# The extracted target is exactly three inversion lengths, so this is the widest
# view available. The displayed window is derived from the alignments below and
# never exceeds it.
region_start <- max(0, inv_start - inv_length)
region_end <- inv_end + inv_length

# Alignments shorter than this are segmental-duplication debris at this scale:
# they add wisps without adding structure. Scaled to the locus, since 2 kb is
# substantial for a 20 kb inversion and noise for a 700 kb one.
min_span <- min(max(0.03 * inv_length, 2000), 25000)
# Flank actually shown on each side: enough aligned sequence to read the
# orientation, and no more.
min_flank_shown <- min(max(0.25 * inv_length, 5000), 50000)

overlap_length <- function(start, end, left, right) {
  pmax(0, pmin(end, right) - pmax(start, left))
}

strand_support <- function(frame, left, right) {
  amount <- overlap_length(frame$t.start, frame$t.end, left, right)
  c(
    forward = sum(amount[frame$strand == "+"], na.rm = TRUE),
    reverse = sum(amount[frame$strand == "-"], na.rm = TRUE)
  )
}

dominant_strand <- function(support) {
  if (sum(support) == 0) {
    return(NA_character_)
  }
  if (support[["forward"]] >= support[["reverse"]]) "+" else "-"
}

paf <- readPaf(
  paf_file,
  include.paf.tags = TRUE,
  restrict.paf.tags = c("cg", "de", "tp")
)
if (nrow(paf) == 0) {
  stop("PAF contains no alignments: ", paf_file)
}

# The target is a region extracted from GRCh38. Restore chromosome coordinates.
paf$t.start <- paf$t.start + region_offset
paf$t.end <- paf$t.end + region_offset

# Divergence, not secondary status, is what separates orthology from the 5-15%
# duplication copies that turn these plots into a haze. Secondary alignments are
# kept: a recurrent inversion is flanked by inverted repeats, so one chimpanzee
# segment matches both flanks and minimap2 demotes one of them: filtering on
# tp:A:P deletes a whole flank at exactly the loci this figure is about.
# A handful of loci sit in sequence where panTro6 has nothing close, so take the
# tightest cut that leaves anything rather than dropping the page, and record it.
raw_alignments <- nrow(paf)
secondary_kept <- sum(paf$tp != "P", na.rm = TRUE)
de_limit <- NA_real_
for (limit in c(0.03, 0.05, 0.08, 0.12)) {
  if (any(paf$de <= limit)) {
    de_limit <- limit
    break
  }
}
if (is.na(de_limit)) {
  stop("No alignment below 12% divergence: ", paf_file)
}
paf <- paf[paf$de <= de_limit, , drop = FALSE]
# Drop short alignments, relaxing the threshold rather than emptying the plot at
# loci whose orthologous blocks are all small.
span_filter <- function(frame, limit) {
  frame[
    (frame$q.end - frame$q.start) >= limit & (frame$t.end - frame$t.start) >= limit,
    ,
    drop = FALSE
  ]
}
while (min_span > 1000 && nrow(span_filter(paf, min_span)) < 2) {
  min_span <- min_span / 2
}
if (nrow(span_filter(paf, min_span)) > 0) {
  paf <- span_filter(paf, min_span)
}

# Choose the chimpanzee sequence with the strongest support across both flanks.
# Prioritizing the weaker flank favors a sequence that spans the complete locus
# rather than a paralog with a strong match on only one side.
contig_scores <- lapply(unique(paf$q.name), function(contig) {
  frame <- paf[paf$q.name == contig, , drop = FALSE]
  left <- sum(overlap_length(
    frame$t.start, frame$t.end, region_start, inv_start
  ))
  right <- sum(overlap_length(
    frame$t.start, frame$t.end, inv_end, region_end
  ))
  interior <- sum(overlap_length(
    frame$t.start, frame$t.end, inv_start, inv_end
  ))
  data.frame(
    contig = contig,
    both_flanks = min(left, right),
    flank_total = left + right,
    window_total = left + interior + right,
    stringsAsFactors = FALSE
  )
})
contig_scores <- do.call(rbind, contig_scores)
# Reaching both flanks is the gate, but ranking on the weaker flank alone picks
# a duplicate that straddles the locus over the sequence that actually covers
# it: at 1q21.1 that preferred an unplaced scaffold covering 60 kb of the
# inversion to chr1 covering 329 kb. Among contigs that reach both flanks, take
# the one with the most aligned sequence across the window.
# Where nothing spans the locus, orientation can only be read from whatever
# flank is aligned, so flank coverage still decides there; total coverage would
# otherwise pick sequence sitting entirely inside the inversion with no anchor.
contig_scores$spans_locus <- contig_scores$both_flanks > 0
contig_scores$spanning_total <- ifelse(
  contig_scores$spans_locus, contig_scores$window_total, 0
)
contig_scores <- contig_scores[
  order(
    -contig_scores$spans_locus,
    -contig_scores$spanning_total,
    -contig_scores$flank_total,
    -contig_scores$window_total,
    contig_scores$contig
  ),
  ,
  drop = FALSE
]
chimp_contig <- contig_scores$contig[[1]]

plot_paf <- paf[paf$q.name == chimp_contig, , drop = FALSE]
plot_paf <- plot_paf[
  plot_paf$t.end > region_start & plot_paf$t.start < region_end,
  ,
  drop = FALSE
]
if (nrow(plot_paf) == 0) {
  stop("Selected chimpanzee contig has no alignment in the plotting window")
}

left_support <- strand_support(plot_paf, region_start, inv_start)
right_support <- strand_support(plot_paf, inv_end, region_end)
combined_support <- left_support + right_support
left_vote <- dominant_strand(left_support)
right_vote <- dominant_strand(right_support)
combined_vote <- dominant_strand(combined_support)
interior_support <- strand_support(plot_paf, inv_start, inv_end)
interior_vote <- dominant_strand(interior_support)

left_rows <- plot_paf[
  overlap_length(plot_paf$t.start, plot_paf$t.end, region_start, inv_start) > 0,
  ,
  drop = FALSE
]
right_rows <- plot_paf[
  overlap_length(plot_paf$t.start, plot_paf$t.end, inv_end, region_end) > 0,
  ,
  drop = FALSE
]
left_gap <- if (nrow(left_rows) > 0) {
  max(0, inv_start - max(left_rows$t.end))
} else {
  Inf
}
right_gap <- if (nrow(right_rows) > 0) {
  max(0, min(right_rows$t.start) - inv_end)
} else {
  Inf
}

if (is.na(combined_vote)) {
  if (is.na(interior_vote)) {
    stop("The selected chimpanzee sequence has no alignment in the plotting window")
  }
  axis_vote <- interior_vote
  axis_rule <- "interior alignment only; no aligned flank sequence"
} else if (!is.na(left_vote) && !is.na(right_vote) && left_vote == right_vote) {
  axis_vote <- left_vote
  axis_rule <- "concordant left and right flanks"
} else if (!is.na(left_vote) && !is.na(right_vote)) {
  # Discordant flanks describe real local rearrangement, not an arbitrary whole-
  # contig orientation. Anchor the display to the flank whose orthologous block
  # reaches closest to its inversion boundary and report the discordance.
  if (left_gap < right_gap) {
    axis_vote <- left_vote
    axis_rule <- "discordant flanks; anchored to boundary-adjacent left flank"
  } else {
    axis_vote <- right_vote
    axis_rule <- "discordant flanks; anchored to boundary-adjacent right flank"
  }
} else if (!is.na(left_vote)) {
  axis_vote <- left_vote
  axis_rule <- "left flank only"
} else {
  axis_vote <- right_vote
  axis_rule <- "right flank only"
}

# A whole chimpanzee chromosome/contig can have arbitrary assembly orientation.
# Reverse its coordinate system and every alignment strand when the flanks vote
# reverse, so the plotted chimpanzee axis always increases in the flank-supported
# direction. This removes meaningless whole-contig flips while preserving a true
# orientation switch inside the inversion.
axis_reversed <- identical(axis_vote, "-")

# Even inside one contig, duplication copies can sit megabases from the locus.
# Left in, the furthest one sets the chimpanzee axis and squashes the locus to a
# sliver. Anchor on the block covering most of the inversion and keep only what
# lies within one region width of it.
inv_covered <- overlap_length(plot_paf$t.start, plot_paf$t.end, inv_start, inv_end)
anchor <- which.max(inv_covered)
query_gap <- pmax(
  plot_paf$q.start - plot_paf$q.end[anchor],
  plot_paf$q.start[anchor] - plot_paf$q.end,
  0
)
clustered <- query_gap <= (region_end - region_start)
dropped_distant <- sum(!clustered)
plot_paf <- plot_paf[clustered, , drop = FALSE]

# Widen from each breakpoint only until min_flank_shown of aligned flank is in
# view, so tight loci are not shown at three times their size and loci whose
# flanking alignment starts far out still show it. The extracted region is the
# ceiling.
extend_side <- function(edge, outward) {
  covered <- 0
  distance <- 0
  limit <- if (outward < 0) edge - region_start else region_end - edge
  step <- max(limit / 200, 1)
  while (covered < min_flank_shown && distance < limit) {
    distance <- min(distance + step, limit)
    lo <- if (outward < 0) edge - distance else edge
    hi <- if (outward < 0) edge else edge + distance
    covered <- sum(overlap_length(plot_paf$t.start, plot_paf$t.end, lo, hi))
  }
  distance
}
window_start <- inv_start - extend_side(inv_start, -1)
window_end <- inv_end + extend_side(inv_end, 1)

# Cut alignments exactly to the displayed human window using their CIGAR
# strings. This keeps both axes and arrows confined to the displayed region.
plot_paf$t.name <- chrom
plot_paf <- subsetPafAlignments(
  paf.table = plot_paf,
  target.region = paste0(chrom, ":", window_start, "-", window_end)
)
if (is.null(plot_paf) || nrow(plot_paf) == 0) {
  stop("No alignments remain after exact window clipping")
}

if (axis_reversed) {
  query_length <- unique(plot_paf$q.len)
  if (length(query_length) != 1 || !is.finite(query_length)) {
    stop("Cannot determine one query length for chimpanzee axis reversal")
  }
  old_start <- plot_paf$q.start
  old_end <- plot_paf$q.end
  plot_paf$q.start <- query_length - old_end
  plot_paf$q.end <- query_length - old_start
  plot_paf$strand <- ifelse(plot_paf$strand == "+", "-", "+")
}

human_name <- "Human (GRCh38)"
chimp_name <- "Chimpanzee (panTro6)"
plot_paf$t.name <- human_name
plot_paf$q.name <- chimp_name

# SVbyEye's default plotMiro coordinate synchronization preserves one base per
# horizontal unit. That is useful for similarly sized sequences, but it can clip
# a chimpanzee endpoint when rearrangements make the aligned chimp span longer
# than the human window. Build the miropeat coordinates explicitly instead:
# human remains in GRCh38 coordinates, while the complete aligned chimp span is
# mapped onto the same panel width and retains its own genomic labels below.
coords <- paf2coords(plot_paf, sync.x.coordinates = FALSE)
query_range <- range(coords$seq.pos[coords$seq.id == "query"])
if (!all(is.finite(query_range)) || diff(query_range) <= 0) {
  stop("Cannot determine a nonzero aligned chimpanzee span")
}
map_query_to_panel <- function(position) {
  window_start +
    (position - query_range[[1]]) *
      (window_end - window_start) / diff(query_range)
}
is_query <- coords$seq.id == "query"
coords$x[is_query] <- map_query_to_panel(coords$seq.pos[is_query])

palette <- c(`+` = "#238b45", `-` = "#4f8ee8")
plot <- ggplot(coords) +
  SVbyEye:::geom_miropeats(
    aes(x = x, y = y, group = group, fill = direction),
    alpha = 0.5
  ) +
  SVbyEye:::geom_miropeats(
    aes(x = x, y = y, group = group),
    fill = NA,
    colour = "grey65",
    linewidth = 0.25
  )

# paf2coords emits a reverse alignment as query start, target end, query end,
# target start. Reading arrow direction back out of that ordering assigns the
# descending pair to the human row, so the reference is drawn reverse and a
# ribbon meets arrows of the opposite colour. Build the arrows from the PAF, so
# each bar carries the alignment's own strand.
arrow_frame <- rbind(
  data.frame(
    start = plot_paf$t.start,
    end = plot_paf$t.end,
    y = 2,
    group = paste0("t", seq_len(nrow(plot_paf))),
    direction = plot_paf$strand
  ),
  data.frame(
    start = map_query_to_panel(plot_paf$q.start),
    end = map_query_to_panel(plot_paf$q.end),
    y = 1,
    group = paste0("q", seq_len(nrow(plot_paf))),
    direction = plot_paf$strand
  )
)

plot <- plot +
  gggenes::geom_gene_arrow(
    data = arrow_frame,
    aes(
      xmin = start,
      xmax = end,
      y = y,
      group = group,
      fill = direction,
      colour = direction
    ),
    arrowhead_height = grid::unit(3, "mm"),
    inherit.aes = FALSE
  ) +
  scale_fill_manual(values = palette, name = "Alignment\ndirection") +
  scale_colour_manual(values = palette, name = "Alignment\ndirection") +
  scale_y_continuous(
    breaks = c(1, 2),
    labels = c(chimp_name, human_name)
  ) +
  ylab(NULL)

# Independent species axes share panel positions but retain genomic coordinates.
query_labels <- pretty(query_range)
query_labels <- query_labels[
  query_labels >= query_range[[1]] & query_labels <= query_range[[2]]
]
query_breaks <- map_query_to_panel(query_labels)
target_labels <- pretty(c(window_start, window_end))
target_labels <- target_labels[
  target_labels >= window_start & target_labels <= window_end
]

plot <- plot +
  scale_x_continuous(
    name = paste0("Chimpanzee (panTro6) ", chimp_contig, " position (bp)"),
    breaks = query_breaks,
    labels = scales::comma(abs(query_labels)),
    sec.axis = sec_axis(
      transform = ~ .,
      name = paste0("Human (GRCh38) ", chrom, " position (bp)"),
      breaks = target_labels,
      labels = scales::comma(target_labels)
    ),
    expand = c(0, 0)
  ) +
  # A human-only box marks the inversion without projecting either human
  # breakpoint onto a possibly non-orthologous chimpanzee base.
  geom_rect(
    data = data.frame(
      xmin = inv_start,
      xmax = inv_end,
      ymin = 1.92,
      ymax = 2.08
    ),
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    colour = "#c51b1d",
    fill = "#ef3b2c",
    alpha = 0.08,
    linewidth = 0.75,
    linetype = "dashed"
  ) +
  coord_cartesian(
    xlim = c(window_start, window_end),
    ylim = c(0.82, 2.18),
    clip = "on"
  ) +
  # Cytoband plus plain coordinates: the internal locus id is not a reader's
  # handle on the region.
  labs(
    title = paste0(
      sub("^chr", "", chrom), label, "  ",
      chrom, ":", scales::comma(inv_start), "-", scales::comma(inv_end)
    )
  ) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line.x = element_line(linewidth = 1),
    axis.ticks.x = element_line(linewidth = 1),
    axis.ticks.length.x = grid::unit(2, "mm"),
    axis.ticks.y = element_blank(),
    plot.title = element_text(size = 24, face = "bold", margin = margin(b = 10)),
    axis.title.x = element_text(size = 18),
    axis.title.x.top = element_text(size = 18, margin = margin(b = 8)),
    axis.text.x = element_text(size = 16),
    axis.text.y = element_text(size = 18, face = "bold"),
    legend.title = element_text(size = 17),
    legend.text = element_text(size = 16),
    legend.key.size = grid::unit(7, "mm"),
    legend.position = "bottom",
    plot.margin = margin(34, 24, 30, 120)
  )

dir.create(dirname(out_prefix), recursive = TRUE, showWarnings = FALSE)
ggsave(
  paste0(out_prefix, ".pdf"),
  plot,
  width = 15,
  height = 6.4,
  limitsize = FALSE
)
ggsave(
  paste0(out_prefix, ".png"),
  plot,
  width = 15,
  height = 6.4,
  dpi = 600,
  limitsize = FALSE
)

orientation <- data.frame(
  inversion = inv_id,
  chimp_contig = chimp_contig,
  window_start = window_start,
  window_end = window_end,
  left_forward_bp = left_support[["forward"]],
  left_reverse_bp = left_support[["reverse"]],
  left_vote = left_vote,
  right_forward_bp = right_support[["forward"]],
  right_reverse_bp = right_support[["reverse"]],
  right_vote = right_vote,
  combined_vote = combined_vote,
  axis_vote = axis_vote,
  axis_rule = axis_rule,
  left_boundary_gap_bp = left_gap,
  right_boundary_gap_bp = right_gap,
  chimp_axis_reversed = axis_reversed,
  raw_alignments = raw_alignments,
  min_span_bp = min_span,
  divergence_limit = de_limit,
  secondary_kept = secondary_kept,
  dropped_distant = dropped_distant,
  alignments_plotted = nrow(plot_paf),
  stringsAsFactors = FALSE
)
write.table(
  orientation,
  paste0(out_prefix, ".orientation.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

cat(
  "wrote", paste0(out_prefix, ".pdf/.png"), "\n",
  "chimp contig:", chimp_contig, "\n",
  "left flank vote:", left_vote, " ",
  paste(names(left_support), left_support, collapse = ", "), "\n",
  "right flank vote:", right_vote, " ",
  paste(names(right_support), right_support, collapse = ", "), "\n",
  "divergence limit:", de_limit, "\n",
  "axis rule:", axis_rule, "\n",
  "chimp axis reversed:", axis_reversed, "\n"
)
