var table = ee.FeatureCollection("projects/salty-cities/assets/police_areas_westerncape");

// Copyright 2024 Google Inc. All Rights Reserved
var imageCollection =
    ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1');
var SCALE_M = 1;  // For larger AOIs reduce at larger scale to avoid OOM issues.

var n = 1;
var nthFeature = ee.Feature(table.toList(table.size()).get(n - 1));
var cdGeocodi = nthFeature.get('Station');
var aoi = nthFeature.geometry();
print('Station Name of Feature' + n + ':', cdGeocodi);



for (var year = 2016; year < 2024; year++) {
  var epoch_s =
      ee.Date(ee.String(year.toString()).cat('-06-30'), 'America/Los_Angeles')
          .millis()
          .divide(1000);
  var mosaic =
      imageCollection.filter(ee.Filter.eq('inference_time_epoch_s', epoch_s))
          .mosaic();
  var count =
      mosaic
          .reduceRegion({
            reducer: ee.Reducer.sum(),
            geometry: aoi,
            scale: SCALE_M,
            crs: aoi.projection(),
            bestEffort: true
          })
          .getNumber('building_fractional_count')
          // Since the pyramiding policy is mean, we need to multiply by
          // (scale_m * 2) ** 2 to recover sum at original 50cm resolution.
          .multiply(ee.Number(SCALE_M * 2).pow(2));
  Map.addLayer(
      mosaic.select('building_presence'), {'min': 0, 'max': 1},
      year.toString());
  print('Building count for year ' + year + ': ', count.getInfo());
}
