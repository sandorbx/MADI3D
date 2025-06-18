# MADI3D
Morphometric Anatomical Data Investigator in Stereographic 3D

VTK and PyQt-based app to visualize, organize, and process 3D data, volumes and surfaces. It supports stereographic 3D visualization and precise segmentation of mesh data, typical use case is  Neuromorphology research, LM to EM visual matching and preparing figures, you can also annotate organize and showcase textured 3d scans. Source code, as well as releases for Mac and Linux, are coming soon.

![interface](https://github.com/sandorbx/MADI/blob/main/MADI-interface.png?raw=true)

## Quickstart Guide

- Drag and drop your data in OBJ, NRRD, Tiff or ZIP (for textured 3d scans package your individual scans in a zip file with the obj and texture files together) format into the render window or the tree widget( rectangular space on the left side of the interface). You can add individual files or whole folders (including multiple folders).

- Within the tree widget, data can be freely organized. Groups and subgroups are supported. Multiple selection is enabled (use Shift+Left Click for range selection and Ctrl+Left Click for selecting multiple items). For annotation, six free-form comment fields are available.

- For volume Data you can adjust the properties through the volume panel, shift the transfer function or add a lower threshold to reveal the higher intensity areas.

- Checkmarks initially load the data. Later, you can control visibility using them. Check state functions are accessible via right-click or from the "Tree" menu.

- You can save and load your project progress. Save files are CSV-based and can be easily edited with Excel.
  
- You can set animation and record and mp4 video through the Animation and recording panel.
