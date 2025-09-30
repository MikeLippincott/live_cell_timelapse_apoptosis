color_palette_dose <- c(
    "0" = "#132B08", #132B08
    "0.61" = "#265A0C", #265A0C
    "1.22" = "#398E0B", #398E0B
    "2.44" = "#4DC507", #4DC507
    "4.88" = "#62FF00", #62FF00
    "9.77" = "#75FF1A", #75FF1A
    "19.53" = "#85FF33", #85FF33
    "39.06" = "#620B8E", #620B8E
    "78.13" = "#410C5A", #410C5A
    "156.25" = "#21082B" #21082B
)

dose_guides_color <- guides(
        color = guide_legend(
            override.aes = list(size = 7, alpha = 1),
            title.position = "top",
            title.hjust = 0.5,
            )
    )
shuffle_guides_shape <- guides(
        shape = guide_legend(
            override.aes = list(size = 7, alpha = 1),
            title.position = "top",
            title.hjust = 0.5,
            nrow = 2,
        )
    )
font_size <- 24
plot_themes <- (
    theme_bw()
    + theme(
        legend.position = "bottom",
        legend.text = element_text(size = font_size),
        legend.title = element_text(size = font_size),
        axis.title.x = element_text(size = font_size),
        axis.text.x = element_text(size = font_size),
        axis.title.y = element_text(size = font_size),
        axis.text.y = element_text(size = font_size),
        strip.text = element_text(size = font_size -2),
    )


)

