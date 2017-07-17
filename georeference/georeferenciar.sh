gdal_translate -of GTiff -gcp 330.018 -157.062 868132 4.04803e+06 -gcp 1003.89 -186.704 869432 4.04817e+06 -gcp 721.297 -323.059 868890 4.04839e+06 -gcp 165.997 -439.652 867766 4.04857e+06 "/home/mario/software/TFM/ejemploGEOR/Georeferencing_Data/puerto_sin_SRC.TIF" "/tmp/puerto_sin_SRC.TIF"
gdalwarp -r near -order 1 -co COMPRESS=NONE  "/tmp/puerto_sin_SRC.TIF" "/home/mario/software/TFM/ejemploGEOR/Georeferencing_Data/puerto_sin_SRC_modificado_corregido.tif"


