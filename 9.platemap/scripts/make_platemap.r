suppressPackageStartupMessages(suppressWarnings(library(ggplot2)))
suppressPackageStartupMessages(suppressWarnings(library(platetools)))
suppressPackageStartupMessages(suppressWarnings(library(RColorBrewer)))
# load theme
source("../../utils/r_themes.r")

# set path to data
data_path <- file.path("..","..","data","platemap_6hr_4ch.csv")
figure_path <- file.path("..","figures")
# make sure the figure path exists
dir.create(figure_path, showWarnings = FALSE)
# figure file name
figure_file <- file.path(figure_path, "platemap_6hr_4ch.png")
# read in data
data <- read.csv(data_path)
head(data)

# format the well column to remove "-"
data$well <- gsub("-", "", data$well)
# make dose categorical
data$dose <- as.factor(data$dose)
# set order of dose levels
unique(data$dose)
data$dose <- factor(data$dose, levels = c(
    "0",
    "0.61",
    "1.22",
    "2.44",
    "4.88",
    "9.77",
    "19.53",
    "39.06",
    "78.13",
    "156.25"
))



width <- 10
height <- 7
options(repr.plot.width = width, repr.plot.height = height)
platemap <- (
    platetools::raw_map(
        data = data$dose,
        well = data$well,
        plate = 96,
        size = 15
    )
    + scale_fill_manual(values = color_palette)
    # change legend title
    + labs(fill = "Dose (nM)")
    # change text size
    + theme(axis.text.x = element_text(size = 18))
    + theme(axis.text.y = element_text(size = 18))
    + theme(legend.text = element_text(size = 18))
    + theme(legend.title = element_text(size = 18, hjust = 0.5))
    # move legend to bottom
    + theme(
        legend.position = "bottom",
    )
    + guides(
        fill = guide_legend(
            nrow = 1,
            byrow = TRUE,
            title.position = "top",
            label.position = "bottom",
            title.hjust = 0.5,
            # rotate the text
            label.theme = element_text(angle = 20, hjust = 0.5, vjust = 0.5),
            # change the size of the dots in the legend
            override.aes = list(size = 14)
        )
    )
)
platemap
# save plot
ggsave(figure_file, platemap, width = width, height = height, units = "in")
