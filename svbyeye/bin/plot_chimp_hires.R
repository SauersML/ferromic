#!/usr/bin/env Rscript

# Publication-resolution human–chimpanzee alignment plot for one inversion.
# Human (GRCh38) is always the upper target track. Chimpanzee (panTro6) is the
# lower query track, reoriented when necessary so the predominant alignment in
# the two flanks is forward. Breakpoint ticks are drawn only beside the human
# track; they do not imply that an exact orthologous chimpanzee breakpoint exists.

suppressPackageStartupMessages({
  library(SVbyEye)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 9) {
  stop(
    "usage: plot_chimp_hires.R PAF OUT_PREFIX INV_ID CHROM INV_START ",
    "INV_END REGION_OFFSET RECURRENCE LABEL"
  )
}

paf_file <- args[[1]]
out_prefix <- args[[2]]
inv_id <- args[[3]]
chrom <- args[[4]]
inv_start <- as.numeric(args[[5]])
inv_end <- as.numeric(args[[6]])
region_offset <- as.numeric(args[[7]])
recurrence <- args[[8]]
label <- args[[9]]

inv_length <- inv_end - inv_start
if (!is.finite(inv_length) || inv_length <= 0) {
  stop("INV_END must be greater than INV_START")
}

# The displayed region is exactly three inversion lengths, centered on the
# inversion: one inversion length of flank on either side.
window_start <- max(0, inv_start - inv_length)
window_end <- inv_end + inv_length

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

paf <- readPaf(paf_file, include.paf.tags = TRUE, restrict.paf.tags = "cg")
if (nrow(paf) == 0) {
  stop("PAF contains no alignments: ", paf_file)
}

# The target is a region extracted from GRCh38. Restore chromosome coordinates.
paf$t.start <- paf$t.start + region_offset
paf$t.end <- paf$t.end + region_offset

# Choose the chimpanzee sequence with the strongest support across both flanks.
# Prioritizing the weaker flank favors a sequence that spans the complete locus
# rather than a paralog with a strong match on only one side.
contig_scores <- lapply(unique(paf$q.name), function(contig) {
  frame <- paf[paf$q.name == contig, , drop = FALSE]
  left <- sum(overlap_length(
    frame$t.start, frame$t.end, window_start, inv_start
  ))
  right <- sum(overlap_length(
    frame$t.start, frame$t.end, inv_end, window_end
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
contig_scores <- contig_scores[
  order(
    -contig_scores$both_flanks,
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
  plot_paf$t.end > window_start & plot_paf$t.start < window_end,
  ,
  drop = FALSE
]
if (nrow(plot_paf) == 0) {
  stop("Selected chimpanzee contig has no alignment in the plotting window")
}

left_support <- strand_support(plot_paf, window_start, inv_start)
right_support <- strand_support(plot_paf, inv_end, window_end)
combined_support <- left_support + right_support
left_vote <- dominant_strand(left_support)
right_vote <- dominant_strand(right_support)
combined_vote <- dominant_strand(combined_support)
if (is.na(combined_vote)) {
  stop("The selected chimpanzee sequence has no aligned bases in either flank")
}

left_rows <- plot_paf[
  overlap_length(plot_paf$t.start, plot_paf$t.end, window_start, inv_start) > 0,
  ,
  drop = FALSE
]
right_rows <- plot_paf[
  overlap_length(plot_paf$t.start, plot_paf$t.end, inv_end, window_end) > 0,
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

if (!is.na(left_vote) && !is.na(right_vote) && left_vote == right_vote) {
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

# Cut alignments exactly to the three-inversion-width human window using their
# CIGAR strings. This keeps both axes and arrows confined to the displayed region.
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

# Arrow coordinates follow the four-row ordering emitted by paf2coords:
# query start, target start, query end, target end for each alignment.
arrow_start <- coords$x[c(TRUE, TRUE, FALSE, FALSE)]
arrow_end <- coords$x[c(FALSE, FALSE, TRUE, TRUE)]
arrow_y <- coords$y[c(TRUE, TRUE, FALSE, FALSE)]
arrow_group <- coords$group[c(TRUE, TRUE, FALSE, FALSE)]
arrow_frame <- data.frame(
  start = arrow_start,
  end = arrow_end,
  y = arrow_y,
  group = arrow_group,
  direction = ifelse(arrow_start < arrow_end, "+", "-")
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
    name = "Chimpanzee (panTro6) genomic position (bp; axis normalized by flanks)",
    breaks = query_breaks,
    labels = scales::comma(abs(query_labels)),
    sec.axis = sec_axis(
      transform = ~ .,
      name = "Human (GRCh38) genomic position (bp)",
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
  labs(
    title = paste0(label, " (", inv_id, ")"),
    subtitle = paste0(
      "Three-inversion-length window; ", recurrence,
      " inversion; aligned panTro6 sequence ", chimp_contig, "."
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
    plot.title = element_text(size = 15, face = "bold"),
    plot.subtitle = element_text(size = 10.5),
    axis.title.x = element_text(size = 10),
    axis.title.x.top = element_text(size = 10),
    axis.text.y = element_text(size = 10, face = "bold"),
    legend.position = "bottom",
    plot.margin = margin(14, 18, 14, 18)
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
