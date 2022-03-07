# Summary

A repository for python scripts that help with modelling in blender

# outline_image.py

This script takes an image of a silhouette and converts it to a 2D object in blender. It works by first outlining the object and then finding the holes within that object. You can see the basic usage with `python outline_image.py -h`. This script will produce a file where the first line contains the vertices of the outline of the object and the subsequent lines contain vertices for the holes within that outline. Once you have generated this file, you can open up blender and go over to the "Scripting" tab. In the console, copy the function in `outline_image.py` called `create_model_from_file`. Then run `create_model_from_file(<path_to_shape_file>, <object_name>)` which will output two objects: one for the outline and another for the holes. From here, you can extrude the objects, recalculate their normals, and subtract out the holes from the outline. I use it to 3D print earrings for my wife :). 

### TODOS
1. Output a single 3D object with the holes removed instead of an outline and a hole object.
2. Turn into a blender addon for ease of use. 
3. Make it work for multiple objects. Currently, this script only works for contiguous objects.
