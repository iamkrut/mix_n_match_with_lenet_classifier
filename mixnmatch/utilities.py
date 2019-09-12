import os
import vtk
from random import randint
from matplotlib import pyplot as plt
from vtk.util import numpy_support
import numpy as np
import scipy.misc
from scipy import ndimage
import scipy.ndimage.interpolation as sni

def append_bounding_boxes(meshes):
    bounding_boxes = []
    for mesh in meshes:     
        bounding_boxes.append(mesh.get_bounding_box())
    #bounding_boxes.append(meshes[0].display_orientation())
    return bounding_boxes

def get_actor(mesh):
    mesh_polydata = mesh.get_polydata()
    appendFilter = vtk.vtkAppendPolyData()
    if vtk.VTK_MAJOR_VERSION <= 5:
        appendFilter.AddInputConnection(mesh_polydata.GetProducerPort())
    else:
        appendFilter.AddInputData(mesh_polydata)

    appendFilter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(appendFilter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def append_meshes(meshes):
    actors = []
    
    for mesh in meshes: 
        actor = get_actor(mesh)
        actors.append(actor)
    
    return actors

def combine_meshes_polydata(meshes_actors):
    
    appendFilter = vtk.vtkAppendPolyData()
    for mesh_actor in meshes_actors:
        mesh_polydata = transformPolyData(mesh_actor)
        if vtk.VTK_MAJOR_VERSION <= 5:
            appendFilter.AddInputConnection(mesh_polydata.GetProducerPort())
        else:
            appendFilter.AddInputData(mesh_polydata)
    appendFilter.Update()

    #  Remove any duplicate points.
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
    cleanFilter.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cleanFilter.GetOutputPort())

    return mapper.GetInputAsDataSet ()

def save_projection(top, left, front, chair_no):
    # Convert result to numpy.
    row, col, _ = top.GetDimensions()
    top = numpy_support.vtk_to_numpy(top.GetPointData().GetScalars())
    top = top.reshape(row, col, -1)
    # top[top == 0] = 255
    # top[top <= 0.3] = 0
    # top[(top > 0.3) & (top < 1)] = 0.6
    # np.extract(top,top)

    left = numpy_support.vtk_to_numpy(left.GetPointData().GetScalars())
    left = left.reshape(row, col, -1)

    front = numpy_support.vtk_to_numpy(front.GetPointData().GetScalars())
    front = front.reshape(row, col, -1)
    # front[front == 0] = 255

    # top
    top = top.squeeze()
    top[top==0] = 255
    top = np.rot90(top)
    top = np.rot90(top)
    top = np.rot90(top)
    top = np.transpose(top)
    fig1 = scipy.misc.toimage(top)
    fig1 = scipy.misc.imresize(fig1, (224, 224))

    # left
    left = left.squeeze()
    left[left==0] = 255
    left = np.rot90(left)
    left = np.rot90(left)
    left = np.rot90(left)
    left = np.transpose(left)
    fig2 = scipy.misc.toimage(left)
    fig2 = scipy.misc.imresize(fig2, (224, 224))
    
    # front
    front = front.squeeze()
    front[front==0] = 255
    front = np.rot90(front)
    front = np.rot90(front)
    front = np.rot90(front)
    front = np.transpose(front)
    fig3 = scipy.misc.toimage(front)
    fig3 = scipy.misc.imresize(fig3, (224, 224))
    
    ctr = 3 * chair_no + 1
    scipy.misc.imsave(os.path.join('Generated_Chairs','{}.bmp'.format(ctr)), fig3)
    scipy.misc.imsave(os.path.join('Generated_Chairs', '{}.bmp').format(ctr + 1), fig2)
    scipy.misc.imsave(os.path.join('Generated_Chairs', '{}.bmp').format(ctr + 2), fig1)

def is_intersect(p0, p1, resolution=256):
    scale0 = p0.scale
    scale1 = p1.scale
    translation0 = p0.translation
    translation1 = p1.translation

    if scale0 > scale1:
        v0 = p0.to_voxel(
            size=resolution * 2, scale=resolution / 2 * scale1 / scale0,
            translation=None)
        v1 = p1.to_voxel(
            size=resolution * 2, scale=resolution / 2,
            translation=(translation1 - translation0) * scale1)
    else:
        v0 = p0.to_voxel(
            size=resolution * 2, scale=resolution / 2,
            translation=None)
        v1 = p1.to_voxel(
            size=resolution * 2, scale=resolution / 2 * scale0 / scale1,
            translation=(translation1 - translation0) * scale1)

    return np.max(v1 * v0) > 0

def transformPolyData(actor):
    polyData = vtk.vtkPolyData()
    polyData.DeepCopy(actor.GetMapper().GetInput())
    transform = vtk.vtkTransform()
    transform.SetMatrix(actor.GetMatrix())
    fil = vtk.vtkTransformPolyDataFilter()
    fil.SetTransform(transform)
    fil.SetInputDataObject(polyData)
    fil.Update()
    polyData.DeepCopy(fil.GetOutput())
    return polyData