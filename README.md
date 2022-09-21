# QuPath's cell detection in Python

Attempting to port QuPath's default cell detection to Python.

The sample image is extracted from the famous CMU-1.svs, part of the OpenSlide freely distributable data (https://openslide.org).

It doesn't perfectly replicate QuPath's method (and the watershed splitting is questionable) but it hopefully gives the general idea.