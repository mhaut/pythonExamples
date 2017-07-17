

    def pixel2Coordinate(self,geotifAddr,pixelPairs):
	# Load the image dataset
	ds = gdal.Open(geotifAddr)
	# Get a geo-transform of the dataset
	gt = ds.GetGeoTransform()
	# Create a spatial reference object for the dataset
	srs = osr.SpatialReference()
	srs.ImportFromWkt(ds.GetProjection())
	# Set up the coordinate transformation object
	srsLatLong = srs.CloneGeogCS()
	ct = osr.CoordinateTransformation(srs,srsLatLong)
	# Go through all the point pairs and translate them to pixel pairings
	latLonPairs = []
	for point in pixelPairs:
		# Translate the pixel pairs into untranslated points
		ulon = point[0][0]*gt[1]+gt[0]
		ulat = point[0][1]*gt[5]+gt[3]
		# Transform the points to the space
		#(lon,lat,holder) = ct.TransformPoint(ulon,ulat)
		# Add the point to our return array
		latLonPairs.append([ulat,ulon])
	return latLonPairs
