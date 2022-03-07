import cv2
import numpy as np

def create_model_from_file(filename, name):
  def create_object(shapes, name='pixels'):
    me = bpy.data.meshes.new(name + "Mesh")
    ob = bpy.data.objects.new(name, me)
    verts = []
    faces = []
    counter = 0
    for n, box in enumerate(shapes):
      face = []
      for i, j in box:
        verts.append((i, j, 0.0))
        face.append(counter)
        counter += 1
      faces.append(tuple(face))
    me.from_pydata(verts, [], faces)
    me.update()
    bpy.context.collection.objects.link(ob)
  infile = open(filename)
  shapes = []
  for line in infile:
      verts = line.rstrip().split(';')
      shape = []
      for v in verts:
          i, j = v.split(',')
          shape.append((float(i), float(j)))
      shapes.append(shape)
  create_object([shapes[0]], name)
  create_object(shapes[1:], 'holes')

def get_binary_image(image, threshold, gt=False, resolution=None):
  if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if resolution is not None:
    image = cv2.resize(image, resolution)
  if gt:
    image = image > threshold
  else:
    image = image < threshold
  return image

def pad_binary_image(image, p=0):
  if p == 0:
    return image
  for i, j in zip(*np.where(image)):
    i_lo = max(0, i - p)
    i_hi = min(image.shape[0], i + p + 1)
    j_lo = max(0, j - p)
    j_hi = min(image.shape[1], j + p + 1)
    image[i_lo:i_hi,j_lo:j_hi] = True
  return image

def clean_binary_image(image, inverse=False):
  kernel = np.array([[0, 1, 0], [1, -10, 1], [0, 1, 0]])
  kernel = kernel / 4
  img = np.logical_not(image.copy()) if inverse else image.copy()
  holes = cv2.filter2D(img.astype(np.float64), -1, kernel) == 1
  img = np.logical_or(img, holes)
  return np.logical_not(img) if inverse else img

def get_outline(image, hole = False):
  for i in range(image.shape[0]):
    if np.any(image[i,:]):
      j = np.min(np.where(image[i,:])[0])
      break
  # first find the x and y derivatives of the image
  Ix = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3, scale=1./8, delta=0, borderType=cv2.BORDER_DEFAULT)
  Iy = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3, scale=1./8, delta=0, borderType=cv2.BORDER_DEFAULT)
  start_idx = (i, j)
  vertices = []
  idx = None
  includes = {}
  if not hole:
    global out_img
    out_img = np.zeros(image.shape)
  while idx != start_idx:
    if idx is None:
      idx = start_idx
    vertices.append(idx)
    if idx != start_idx:
      includes[idx] = True
    out_img[idx] = 1
    cv2.imshow('out', out_img)
    cv2.waitKey(1)
    ix = Ix[idx]
    iy = Iy[idx]
    best_val = 0
    best_tie = 0
    next_idx = None
    y, x = idx
    for dx, dy in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
      # find the direction (up, down, right, or left) that is most in line
      # with the gradient of the image
      if image[y + dy, x + dx] and (y + dy, x + dx) not in includes:
        val = dx * ix - dy * iy
        tie = abs(Ix[y + dy, x + dx]) + abs(Iy[y + dy, x + dx])
        if val > best_val:
          best_val = val
          best_tie = tie
          next_idx = (y + dy, x + dx)
        elif val == best_val:
          if tie > best_tie:
            best_tie = tie
            next_idx = (y + dy, x + dx)
    if next_idx is None:
      break
    idx = next_idx
  return Shape(vertices)

def get_holes(image, outline):
  negative = np.logical_not(image)
  interior = outline.fill()
  for i in range(negative.shape[0]):
    for j in range(negative.shape[1]):
      if (i, j) not in interior:
        negative[i, j] = False
  # add padding of 1 to ensure all holes enclose at least 1 pixel
  negative = pad_binary_image(negative, 1)
  holes = []
  while np.any(negative):
    hole = get_outline(negative, True)
    hole_interior = hole.fill()
    for i, j in hole_interior:
      negative[i,j] = False
    negative = clean_binary_image(negative, inverse=True)
    holes.append(hole)
  return holes

def output_shape(image, threshold, filename, gt=False, resolution=None, padding=1):
  # convert the image to binary
  image = get_binary_image(image, threshold, gt, resolution)

  # pad the object to make it thicker
  image = pad_binary_image(image, padding)

  # get rid of isolated pixels
  image = clean_binary_image(image)

  # find the outline of the object
  outline = get_outline(image)

  # find the holes withing the object
  holes = get_holes(image, outline)
  outfile = open(filename, 'w')
  outfile.write(str(outline) + '\n')
  for h in holes:
    outfile.write(str(h) + '\n')
  outfile.close()

def isbetween(val, c1, c2):
  if c1 < c2:
    return c1 < val <= c2
  return c2 <= val < c1

class Edge():
  def __init__(self, v1, v2):
    self.v1 = v1
    self.v2 = v2
    # axis is the axis that is shared between the vertices
    self.axis = int(v1[1] == v2[1])
    self.off_axis = (self.axis + 1) % 2
    self.direction = np.array([0,0])
    self.direction[self.off_axis] = 1 if self.v2[self.off_axis] > self.v1[self.off_axis] else -1

  def __eq__(self, other):
    return self.v1 == other.v1 and self.v2 == other.v2

  def __repr__(self):
    return 'Edge ({},{})<->({},{})'.format(*self.v1, *self.v2)

class Shape():
  def __init__(self, vertices):
    self.vertices = vertices
    self.edges = {}
    n = len(vertices)
    for i, v in enumerate(vertices):
      self.edges[v] = Edge(v, vertices[(i+1)%n])
    self.lows = np.min(self.vertices, axis=0)
    self.highs = np.max(self.vertices, axis=0)

  def plot(self, shape=None):
    x = [v[0] for v in self.vertices] + [self.vertices[0][0]]
    y = [v[1] for v in self.vertices] + [self.vertices[0][1]]
    plt.plot(x, y)
    if shape is not None:
      x2 = [v[0] for v in shape.vertices] + [shape.vertices[0][0]]
      y2 = [v[1] for v in shape.vertices] + [shape.vertices[0][1]]
      plt.plot(x2, y2, color='r')

  def fill(self):
    """
    Get the indices of all the pixels interior to this shape 
    Since vertices are listed in clockwise order, if you turn to the right
    along the current edge, that should be interior to the shape, so move in 
    that direction until you hit another vertex
    """
    # rotation matrix for 90 degrees to the right
    R = np.array([[0, 1], [-1, 0]])
    interior = set()
    for v in self.vertices:
      interior.add(v)
      e = self.edges[v]
      dx, dy = np.matmul(R, e.direction)
      vertex = (v[0] + dx, v[1] + dy)
      inbounds = True
      additions = []
      while vertex not in self.edges:
        if not self.inbounds(vertex):
          inbounds = False
          break
        additions.append(vertex)
        vertex = (vertex[0] + dx, vertex[1] + dy)
      vertex = (e.v2[0] + dx, e.v2[1] + dy)
      while vertex not in self.edges:
        if not self.inbounds(vertex):
          inbounds = False
          break
        additions.append(vertex)
        vertex = (vertex[0] + dx, vertex[1] + dy)
      if inbounds:
        for vertex in additions:
          interior.add(vertex)
    return interior

  def inbounds(self, vertex):
    return self.lows[0] <= vertex[0] <= self.highs[0] and self.lows[1] <= vertex[1] <= self.highs[1]

  def __str__(self):
    return ';'.join(['{},{}'.format(*v) for v in self.vertices])

  def __lt__(self, other):
    return len(self) < len(other)

  def __len__(self):
    return len(self.vertices)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Convert 2D image into 3D model')

  parser.add_argument('-i', '--input_file', dest='input_file',
    help='path to input image')
  parser.add_argument('-o', '--output_file', dest='output_file', default='./shape.csv',
    help='path to output file with shape data')
  parser.add_argument('-t', '--threshold', dest='threshold', type=int, default=127,
    help='pixel threshol that object is either above or below (see light_obj argument)')
  parser.add_argument('-gt', '--light_obj', dest='light_obj', action='store_true', default=False,
    help='Pass this flag if you have a light object on a dark background')
  parser.add_argument('-r', '--resolution', dest='resolution', default=None,
    help='Desired resolution of 3D object. Defaults to resolution of input image')
  parser.add_argument('-p', '--padding', dest='padding', type=int, default=0,
    help='Number of pixels to pad object')

  args = parser.parse_args()

  if args.resolution is not None:
    try:
      h, w = args.resolution.split(',')
      args.resolution = (int(h), int(w))
    except ValueError:
      raise ValueError("resolution must be two comma separated integers (e.g. 1000,1000)")
    

  image = cv2.imread(args.input_file)
  output_shape(
    image
    , args.threshold
    , args.output_file
    , args.light_obj
    , args.resolution
    , args.padding
  )

  
