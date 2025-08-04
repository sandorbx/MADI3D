# MADI3D

<img src="https://github.com/sandorbx/MADI/blob/main/Madi_splashscreen.png?raw=true"
     alt="splash"
     width="50%"/>

    
Morphometric Anatomical Data Investigator with Stereographic 3D

*Now you can drag and drop Neuronbridge Color MIP search results (https://neuronbridge.janelia.org/), downloadable in csv format, matched EM neurons and LM would be fetched automatically through the Neuronbridge API.

VTK and PyQt-based app to visualize, organize, and process 3D data: fluorescenct microscopy stacks, volumes and surfaces. It supports stereographic 3D visualization and precise segmentation of mesh data. Rendering of fluorescence microscopy stacks are especially accurate because MADI3D uses a similar method descriped in this paper: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1694-9#Sec2.  Typical use case is  Neuromorphology research, LM to EM visual matching and preparing figures, you can also annotate organize and showcase textured 3d scans. Source code, as well as releases for Mac and Linux, are coming soon.

![interface](https://github.com/sandorbx/MADI/blob/main/MADI3D_02.png?raw=true)

![interface](https://github.com/sandorbx/MADI/blob/main/MADI-interface.png?raw=true)

![interface](https://github.com/sandorbx/MADI/blob/main/MADI3D_03.png?raw=true)

## Quickstart Guide

- Download the latest release : https://github.com/sandorbx/MADI3D/releases/download/v25.092/MADI3D_v26.10.zip , unzip in a folder and run the exe file.

- Drag and drop your data in OBJ, SWC, NRRD, Tiff, H5j or ZIP (for textured 3d scans package your individual scans in a zip file with the obj and texture files together) format into the render window or the tree widget( rectangular space on the left side of the interface). You can add individual files or whole folders (including multiple folders).

- Within the tree widget, data can be freely organized. Groups and subgroups are supported. Multiple selection is enabled (use Shift+Left Click for range selection and Ctrl+Left Click for selecting multiple items). For annotation, six free-form comment fields are available.

- Functions for mesh or volume operations and data management are on the right click menu

- For volume Data you can adjust the properties through the volume panel, shift the transfer function or add a lower threshold to reveal the higher intensity areas.

- Checkmarks initially load the data. Later, you can control visibility using them. Check state functions are accessible via right-click or from the "Tree" menu.

- You can save and load your project progress. Save files are CSV-based and can be easily edited with Excel.
  
- You can set animation and record and mp4 video through the Animation and recording panel.
