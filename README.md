# Antarctic-Fracture-Mapping
## Code for creating fracture maps over the Antarctic Ice Sheet from Level-0 IW SLC data from Sentinel-1.
### We:
      - Download data
      - Create radiometrically terrain corrected SAR backscatter images at 50m resolution
      - Create type-A and -B fracture maps for these
      - Create monthly fracture mosaics
--- more info to come ---

There is not yet a great deal of documentation for this code, and the commenting is pretty sparse.
Provided in the directories in this repo are scripts for carrying out prcocessing (generally in Python), and example shell scripts that set up processing on an HPC cluster. Scripts for creating the SAR backscatter images are written in bash and call functions from the GAMMA Remote Sensing software (https://www.gamma-rs.ch/software) which is not freely available - sorry about that.
The Python code is written for the most part in a declarative functional style nicely suited to pipelining data. This should make the scripts a bit easier to make sense of, but could also be confusing (I'm thinking particularly of the scripts for the production of fracture maps from SAR images contained in ./create_fracture_maps/python_scripts/). Feel free to contact me.
