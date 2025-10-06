libraries <- c("ggplot2", "arrow", "tidyr", "stringr", "ggmagnify")
for (lib in libraries) {
    if ("ggmagnify" == lib) {
        if (!requireNamespace("ggmagnify", quietly = TRUE)) {
            install.packages("remotes")
            remotes::install_github("hughjonesd/ggmagnify")
        }
    }
    suppressPackageStartupMessages(
        suppressWarnings(
            library(lib, character.only = TRUE)
        )
    )
}
source("../../utils/r_themes.r")

offset_df <- arrow::read_parquet("../results/all_offset_results.parquet")
metadata_df <- read.csv("../../data/platemap_6hr_4ch.csv")
figures_dir <- "../figures/offsets/"
if (!dir.exists(figures_dir)) {
  dir.create(figures_dir)
}
# get the well from the wellfov
split_columns <- str_split_fixed(offset_df$well_fov, "_", 2)
offset_df$well <- split_columns[, 1]
# merge the metadata with the offset_df on the well column
offset_df <- merge(offset_df, metadata_df, by = "well")

offset_df$dose <- as.character(offset_df$dose)
offset_df$dose <- factor(
    offset_df$dose,
    levels = c(
        '0',
        '0.61',
        '1.22',
        '2.44',
        '4.88',
        '9.77',
        '19.53',
        '39.06',
        '78.13',
        '156.25'
    )
)
unique(offset_df$dose)

# plotting image line
# where the line forms a box
# -950 to 950
box_coordinates <- data.frame(
  x = c(-950, 950, 950, -950, -950),
  y = c(-950, -950, 950, 950, -950)
)

head(offset_df)

width <- 8
height <- 8
dpi <- 600
options(repr.plot.width=width, repr.plot.height=height)
# plot the results
offsets_plot <- (
    ggplot(offset_df, aes(x = x_offset, y = y_offset, color= dose))
    + geom_point(aes(color = dose), alpha = 0.6, size = 2)
    + theme_bw()
    + labs(
        x = "X Offset (pixels)",
        y = "Y Offset (pixels)",
        color = "Stuarosporine dose (nM)"
    )
    + scale_color_manual(values = color_palette_dose)

    + theme(
        legend.position = "right",
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16)
    )
    # make points in legend alpha 1
    + guides(color = guide_legend(override.aes = list(alpha = 1, size = 5)))
    # plot the box
    + geom_polygon(data = box_coordinates, aes(x = x, y = y), fill = NA, color = "black", size = 1)
    + ggplot2::coord_fixed(ratio = 1)
)

from <- c(
    xmin = -200,
    xmax = 0,
    ymin = -100,
    ymax = 50
)
to <- c(
    xmin = -150,
    xmax = 900,
    ymin = 100,
    ymax = 900
)
# place the zoomed in plot on the main plot
offsets_plot_with_zoom <- (
    offsets_plot
    + geom_magnify(
        from = from,
        to = to,
        size = 0.25,
        fill = "white",
        color = "black",
        expand = 0.02
    )
)
ggsave(
    offsets_plot_with_zoom,
    filename = "../figures/offsets_plot.png",
    width = 8,
    height = 8,
    dpi = 300
)
offsets_plot_with_zoom
