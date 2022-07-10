# About
testing some code examples from [some dude's blog post](https://ai-facets.org/robust-logo-detection-with-opencv/) and [opencv docs](https://ai-facets.org/robust-logo-detection-with-opencv/)

The only real test already tried is on the verizon screenshot and logo in `images` dir. 
The objective is to utilize an opencv algorithm to make a detector that trains on one logo and detects it in a screenshot. Should work on the rest of the screenshots (with logos to be downloaded) as well.

# Installation
```
pip install -r requirements.txt
```

# Running examples
```
python orb_test.py
python sift_test.py
```