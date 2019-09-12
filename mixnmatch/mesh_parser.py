import vtk
from vtk.util import numpy_support
import numpy as np
from numpy import linalg
from sklearn.decomposition import PCA
import warnings

class MeshParser:

    def loadObj(fname):
        """Load the given STL file, and return a vtkPolyData object for it."""
        reader = vtk.vtkOBJReader()
        reader.SetFileName(fname)
        reader.Update()
        polydata = reader.GetOutput()
        return polydata

    def get_relative_scale(self):
        return self.scale

    def get_relative_translation(self):
        return self.translation

    def __init__(self, part_path=None, chair_path=None, poly_data=None, scale=None, translation=None):
        
        self.part_path = part_path
        self.chair_path = chair_path
        self.is_axes_enabled = True
        self.is_bounding_box_enabled = True
        self.is_orientation_enabled = False
        self.is_normalized = False
        self.is_debug = False

        self.voxel = None
        self.bounds = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None
        self.translation = None
        self.scale = None

        if(poly_data == None):
            
            # read part
            reader = vtk.vtkOBJReader()
            reader.SetFileName(self.part_path)
            reader.Update()
            self.poly_data = reader.GetOutput()
            self.points = self.poly_data.GetPoints()
            self.polys = self.poly_data.GetPolys()

            # read chair
            reader = vtk.vtkOBJReader()
            reader.SetFileName(self.chair_path)
            reader.Update()
            self.chair_poly_data = reader.GetOutput()

            self.__set()
            self.normalize()
        else:
            self.poly_data = poly_data
            self.points = self.poly_data.GetPoints()
            self.polys = self.poly_data.GetPolys()
            self.bounds = self.points.GetBounds()
            self.x_min = self.bounds[0]
            self.x_max = self.bounds[1]
            self.y_min = self.bounds[2]
            self.y_max = self.bounds[3]
            self.z_min = self.bounds[4]
            self.z_max = self.bounds[5]

            self.scale = scale
            self.translation = translation

    def __set(self):
        self.points = self.poly_data.GetPoints()
        self.polys = self.poly_data.GetPolys()
        self.bounds = self.points.GetBounds()
        self.x_min = self.bounds[0]
        self.x_max = self.bounds[1]
        self.y_min = self.bounds[2]
        self.y_max = self.bounds[3]
        self.z_min = self.bounds[4]
        self.z_max = self.bounds[5]

        self.chair_bounds = self.chair_poly_data.GetPoints().GetBounds()
        self.chair_x_min = self.chair_bounds[0]
        self.chair_x_max = self.chair_bounds[1]
        self.chair_y_min = self.chair_bounds[2]
        self.chair_y_max = self.chair_bounds[3]
        self.chair_z_min = self.chair_bounds[4]
        self.chair_z_max = self.chair_bounds[5]

    def normalize(self):
        if not self.is_normalized:

            # Relocate to center.
            if self.is_debug:
                print('Bounding Before Centralization')
                print(self.bounds, '\n')

            transform = vtk.vtkTransform()
            self.translation = np.zeros(3)
            self.translation[0] = (self.chair_x_min + self.chair_x_max) / 2
            self.translation[1] = (self.chair_y_min + self.chair_y_max) / 2
            self.translation[2] = (self.chair_z_min + self.chair_z_max) / 2
            transform.Translate(
                -self.translation[0],
                -self.translation[1],
                -self.translation[2])

            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetTransform(transform)
            transform_filter.SetInputData(self.poly_data)
            transform_filter.Update()
            self.poly_data = transform_filter.GetOutput()
            self.__set()

            if self.is_debug:
                print('Bounding After Centralization')
                print(self.bounds, '\n')

            # Normalize the scale.
            transform = vtk.vtkTransform()
            self.scale = 1 / np.amax(np.absolute(self.chair_bounds))
            transform.Scale(self.scale, self.scale, self.scale)
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetTransform(transform)
            transform_filter.SetInputData(self.poly_data)
            transform_filter.Update()
            self.poly_data = transform_filter.GetOutput()
            self.__set()

            self.is_normalized = True

            if self.is_debug:
                print('Bounding After Normalization')
                print(self.bounds, '\n')

    def __display_axes(self):
        transform = vtk.vtkTransform()
        boundaries = [
            abs(self.x_min),
            abs(self.x_max),
            abs(self.y_min),
            abs(self.y_max),
            abs(self.z_min),
            abs(self.z_max)]
        scale = max(boundaries) * 1.1
        if self.is_normalized:
            scale = 5 / self.scale
        transform.Scale(scale, scale, scale)
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)
        return axes

    def get_top_center_point(self):
        return np.array([(self.x_min + self.x_max) / 2, (self.y_min + self.y_max)/2, self.z_max])

    def get_bottom_center_point(self):
        return np.array([(self.x_min + self.x_max) / 2, (self.y_min + self.y_max)/2, self.z_min])

    def get_center_back_top_point(self):
        return np.array([(self.x_min + self.x_max) / 2, self.y_max, self.z_max])

    def get_center_back_bottom_point(self):
        return np.array([(self.x_min + self.x_max) / 2, self.y_max, self.z_min])

    def get_center_point(self):
        return np.array([(self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2, (self.z_min + self.z_max)/2])

    def get_center_front_right_corner(self):
        p4 = np.array([self.x_min, self.y_min, self.z_min])
        return p4
    
    def get_center_front_left_corner(self):
        p7 = np.array([self.x_max, self.y_min, self.z_min])
        return p7

    def get_center_back_left_corner(self):
        p6 = np.array([self.x_max, self.y_max, self.z_min])
        return p6
    
    def get_center_back_right_corner(self):
        p5 = np.array([self.x_min, self.y_max, self.z_min])
        return p5
    
    def get_scale(self):
        return np.array([abs(self.x_min - self.x_max), abs(self.y_min - self.y_max), abs(self.z_min - self.z_max)])

    #   1----2  Y
    #  /|   /|  ^
    # 0-5--3-6  |
    # |/   |/   |
    # 4----7    Z------->X
    def get_bounding_box(self):
        p0 = [self.x_min, self.y_max, self.z_max]
        p1 = [self.x_min, self.y_max, self.z_min]
        p2 = [self.x_max, self.y_max, self.z_min]
        p3 = [self.x_max, self.y_max, self.z_max]
        p4 = [self.x_min, self.y_min, self.z_max]
        p5 = [self.x_min, self.y_min, self.z_min]
        p6 = [self.x_max, self.y_min, self.z_min]
        p7 = [self.x_max, self.y_min, self.z_max]

        points = vtk.vtkPoints()
        points.InsertNextPoint(p0)
        points.InsertNextPoint(p1)
        points.InsertNextPoint(p2)
        points.InsertNextPoint(p3)
        points.InsertNextPoint(p4)
        points.InsertNextPoint(p5)
        points.InsertNextPoint(p6)
        points.InsertNextPoint(p7)

        line00 = vtk.vtkLine()
        line00.GetPointIds().SetId(0, 0)
        line00.GetPointIds().SetId(1, 1)
        line01 = vtk.vtkLine()
        line01.GetPointIds().SetId(0, 1)
        line01.GetPointIds().SetId(1, 2)
        line02 = vtk.vtkLine()
        line02.GetPointIds().SetId(0, 2)
        line02.GetPointIds().SetId(1, 3)
        line03 = vtk.vtkLine()
        line03.GetPointIds().SetId(0, 3)
        line03.GetPointIds().SetId(1, 0)
        line04 = vtk.vtkLine()
        line04.GetPointIds().SetId(0, 0)
        line04.GetPointIds().SetId(1, 4)
        line05 = vtk.vtkLine()
        line05.GetPointIds().SetId(0, 1)
        line05.GetPointIds().SetId(1, 5)
        line06 = vtk.vtkLine()
        line06.GetPointIds().SetId(0, 2)
        line06.GetPointIds().SetId(1, 6)
        line07 = vtk.vtkLine()
        line07.GetPointIds().SetId(0, 3)
        line07.GetPointIds().SetId(1, 7)
        line08 = vtk.vtkLine()
        line08.GetPointIds().SetId(0, 4)
        line08.GetPointIds().SetId(1, 5)
        line09 = vtk.vtkLine()
        line09.GetPointIds().SetId(0, 5)
        line09.GetPointIds().SetId(1, 6)
        line10 = vtk.vtkLine()
        line10.GetPointIds().SetId(0, 6)
        line10.GetPointIds().SetId(1, 7)
        line11 = vtk.vtkLine()
        line11.GetPointIds().SetId(0, 7)
        line11.GetPointIds().SetId(1, 4)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line00)
        lines.InsertNextCell(line01)
        lines.InsertNextCell(line02)
        lines.InsertNextCell(line03)
        lines.InsertNextCell(line04)
        lines.InsertNextCell(line05)
        lines.InsertNextCell(line06)
        lines.InsertNextCell(line07)
        lines.InsertNextCell(line08)
        lines.InsertNextCell(line09)
        lines.InsertNextCell(line10)
        lines.InsertNextCell(line11)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def display_orientation(self, num_component=3):
        self.voxel = self.to_voxel(256)
        pca = PCA(n_components=num_component)
        pca = pca.fit(np.argwhere(self.voxel))
        components = np.asarray(pca.components_)
        components = components / linalg.norm(
            components, ord=2, axis=1, keepdims=True)

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        points.InsertNextPoint([0, 0, 0])
        for i in range(num_component):
            points.InsertNextPoint(
                [components[i][0], components[i][1], components[i][2]])

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, i)
            lines.InsertNextCell(line)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)

        transform = vtk.vtkTransform()
        transform.Scale(self.scale, self.scale, self.scale)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(poly_data)
        transform_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def display(self):
        background_color = [0.0, 0.0, 0.0]

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(self.points)
        poly_data.SetPolys(self.polys)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create a rendering window and renderer.
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(background_color)
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)

        # Create a render window interactor.
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        # Assign actor to the renderer.
        renderer.AddActor(actor)
        if self.is_axes_enabled:
            renderer.AddActor(self.__display_axes())
        if self.is_bounding_box_enabled:
            renderer.AddActor(self.get_bounding_box())
        if self.is_orientation_enabled:
            renderer.AddActor(self.__display_orientation())

        # Enable user interface interactor.
        render_window_interactor.Initialize()
        render_window.Render()
        render_window_interactor.Start()

    def get_polydata(self):
        return self.poly_data

    def set_is_normalized(self):
        self.is_normalized = True

    def to_voxel(self, size, scale=None, translation=None):
        if self.voxel is not None and self.voxel.shape[0] == size:
            return self.voxel
        voxel = np.zeros((size, size, size))

        # Scale the mesh so it will span more space when converted to voxel.
        transform = vtk.vtkTransform()
        s = 1
        if self.is_normalized and scale is None:
            s = size / 2
        elif scale is not None:
            s = scale
        transform.Scale(s, s, s)

        # Translate the poly data if needed.
        if translation is not None:
            transform.Translate(
                translation[0],
                translation[1],
                translation[2])
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(self.poly_data)
        transform_filter.Update()

        # Create the background image.
        background = vtk.vtkImageData()
        data_matrix = np.ones([size, size])
        data_matrix = data_matrix * 255
        depth_array = numpy_support.numpy_to_vtk(
            data_matrix.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        background.SetDimensions(size, size, 1)
        background.SetSpacing(1, 1, 1,)
        background.SetOrigin(0, 0, 0)
        background.GetPointData().SetScalars(depth_array)

        # Creating cutting plane for layered cross section.
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, 0)
        plane.SetNormal(0, 0, 1)

        # Shift the model so its bottom is cut first.
        poly_data = transform_filter.GetOutput()
        translate = vtk.vtkTransform()
        translate.Translate(0, 0, size / 2)
        translate_filter = vtk.vtkTransformPolyDataFilter()
        translate_filter.SetTransform(translate)
        translate_filter.SetInputData(poly_data)
        translate_filter.Update()
        poly_data = translate_filter.GetOutput()

        shift = vtk.vtkTransform()
        shift.Translate(0, 0, -1)

        for level in range(size):

            # Initialize cutter based on the plane.
            cutter = vtk.vtkCutter()
            cutter.SetInputData(poly_data)
            cutter.SetCutFunction(plane)

            # Create stripper.
            stripper = vtk.vtkStripper()
            stripper.SetInputConnection(cutter.GetOutputPort())

            # Convert processed poly data to image stencil.
            poly_data_to_stencil = vtk.vtkPolyDataToImageStencil()
            poly_data_to_stencil.SetInputConnection(stripper.GetOutputPort())
            poly_data_to_stencil.SetOutputOrigin(-size / 2, -size / 2, 0.0)

            # Apply stencil.
            stencil = vtk.vtkImageStencil()
            stencil.SetInputData(background)
            stencil.SetStencilConnection(poly_data_to_stencil.GetOutputPort())
            stencil.Update()
            cross_section = stencil.GetOutput()

            # Convert result to numpy.
            row, col, _ = cross_section.GetDimensions()
            scaler = cross_section.GetPointData().GetScalars()
            cross_section = numpy_support.vtk_to_numpy(scaler)
            cross_section = cross_section.reshape(row, col, -1)

            # Assign voxel data.
            cross_section[cross_section == 1] = 0
            cross_section[cross_section == 255] = 1
            cross_section = np.squeeze(cross_section)
            voxel[level] = cross_section

            # Update poly data.
            shift_filter = vtk.vtkTransformPolyDataFilter()
            shift_filter.SetTransform(shift)
            shift_filter.SetInputData(poly_data)
            shift_filter.Update()
            poly_data = shift_filter.GetOutput()

        # Rotate voxel to correct orientation.
        voxel = np.transpose(voxel)
        self.voxel = voxel

        return voxel

    def rotate(self, angle, axis):
        transform = vtk.vtkTransform()
        if (axis == 0):
            transform.RotateWXYZ(angle,1,0,0)
        if axis == 1:
            transform.RotateWXYZ(angle,0,1,0)
        if axis == 2:
            transform.RotateWXYZ(angle,0,0,1)
        transformFilter=vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(self.poly_data)
        transformFilter.Update()

        self.poly_data = transformFilter.GetOutput()
        self.points = self.poly_data.GetPoints()
        self.polys = self.poly_data.GetPolys()
        self.bounds = self.points.GetBounds()
        self.x_min = self.bounds[0]
        self.x_max = self.bounds[1]
        self.y_min = self.bounds[2]
        self.y_max = self.bounds[3]
        self.z_min = self.bounds[4]
        self.z_max = self.bounds[5]

    def get_projections(self, size):

        # Initialize 3 projection images.
        top_img = vtk.vtkImageData()
        top_img.SetDimensions(size, size, 1)
        top_img.SetSpacing(1, 1, 1,)
        top_img.SetOrigin(0, 0, 0)

        left_img = vtk.vtkImageData()
        left_img.SetDimensions(size, size, 1)
        left_img.SetSpacing(1, 1, 1,)
        left_img.SetOrigin(0, 0, 0)

        front_img = vtk.vtkImageData()
        front_img.SetDimensions(size, size, 1)
        front_img.SetSpacing(1, 1, 1,)
        front_img.SetOrigin(0, 0, 0)

        # Stack Matrix
        # Plane [0 0 0]  Plane [0 1 0]  Plane [0 2 0] ...
        #   0 - 0          1 - 1          2 - 2
        #  /   /          /   /          /   /
        # 0 - 0          1 - 1          2 - 2
        stack_matrix = np.arange(size)
        stack_matrix = np.repeat(stack_matrix, size * size)
        stack_matrix = stack_matrix.reshape((size, size, size))
        stack_matrix = np.rot90(stack_matrix)

        # Compute front view.
        top_voxel = self.to_voxel(size)
        top_matrix = stack_matrix * top_voxel
        top_matrix = np.amax(top_matrix, axis=1)
        top_matrix = np.rot90(top_matrix)

        # Compute left view.
        left_voxel = top_voxel
        left_matrix = np.rot90(stack_matrix) * left_voxel
        left_matrix = np.amax(left_matrix, axis=0)

        # Compute top view.
        front_voxel = top_voxel
        front_matrix = np.rot90(stack_matrix, axes=(1, 2)) * front_voxel
        front_matrix = np.amax(front_matrix, axis=2)
        front_matrix = np.rot90(front_matrix)
        front_matrix = np.rot90(front_matrix)
        front_matrix = np.rot90(front_matrix)
        front_matrix = np.flip(front_matrix, axis=1)

        # Assign scalar to images.
        depth_array = numpy_support.numpy_to_vtk(
            top_matrix.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        top_img.GetPointData().SetScalars(depth_array)
        depth_array = numpy_support.numpy_to_vtk(
            left_matrix.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        left_img.GetPointData().SetScalars(depth_array)
        depth_array = numpy_support.numpy_to_vtk(
            front_matrix.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        front_img.GetPointData().SetScalars(depth_array)

        return top_img, left_img, front_img


# p = Mesh("Dataset/Chair_2/chair.obj")
# p.normalize()
# p.display()
